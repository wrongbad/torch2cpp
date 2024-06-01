import torch
import operator
import struct


def shape_dtype(shape, ref=False, const=False):
    shape = ','.join(str(d) for d in shape)
    dtype = f'ml::tensor<{shape}>'
    if ref:
        dtype += '&'
    if const:
        dtype = 'const '+dtype
    return dtype

def val_dtype(val, ref=False, const=False):
    if isinstance(val, (list, tuple)):
        subtypes = [val_dtype(a, ref, const) for a in val]
        subtypes = ','.join(subtypes)
        return f'std::tuple<{subtypes}>'
    return shape_dtype(tuple(val.shape), ref, const)

def flatten(val):
    if isinstance(val, (list, tuple)):
        return [a for vx in val for a in flatten(vx)]
    return [val]


class CustomTracer(torch.fx.Tracer):
    def is_leaf_module(self, mod, name):
        # Tracing into Modules like torch.nn.Linear
        # makes translation easier
        return False


    
class Interpreter(torch.fx.Interpreter):
    inputs = {}
    output_type = None
    output_ref = None
    weights = {}

    tmp_vars = {}
    node_vars = {}

    fwds = []

    def get_tmp(self, node, shape):
        refcount = len(node.users)
        for name, info in self.tmp_vars.items():
            tshape, tref = info
            if shape == tshape and tref == 0:
                self.node_vars[node] = name
                info[1] = refcount
                return name
        name = f'tmp{len(self.tmp_vars)}'
        self.tmp_vars[name] = [shape, refcount]
        self.node_vars[node] = name
        return name

    def deref(self, node):
        if node is None:
            return 'nullptr'
        if isinstance(node, slice):
            if node == slice(None, None, None):
                return 'slice<>()'
            else:
                raise ValueError('unsupported '+str(node))
        if not isinstance(node, torch.fx.Node):
            return str(node)
        if node not in self.node_vars:
            return node.name
        name = self.node_vars[node]
        if name in self.tmp_vars:
            self.tmp_vars[name][1] -= 1
            assert self.tmp_vars[name][1] >= 0
        return name

    def nested_refstr(self, arg):
        if isinstance(arg, (list, tuple)):
            nest = [self.nested_refstr(a) for a in arg]
            return '{'+', '.join(nest)+'}'
        return self.node_vars[arg]

    def alias(self, node, src):
        src = self.node_vars[src]
        self.node_vars[node] = src
        self.tmp_vars[src][1] += len(node.users) - 1


    def run_node(self, n):
        with self._set_current_node(n):

            args, kwargs = self.fetch_args_kwargs_from_env(n)
            val = getattr(self, n.op)(n.target, args, kwargs)

            if n.op == 'placeholder':
                self.inputs[n.name] = val_dtype(val)
                self.node_vars[n] = n.name
            elif n.op == 'get_attr':
                self.weights[n.name] = (val_dtype(val, const=True), val)
                self.node_vars[n] = n.name
            elif n.op == 'call_function' or n.op == 'call_method':

                fname = n.target
                if 'fun' in n.op:
                    fname = fname.__name__

                if n.target == operator.getitem:
                    if isinstance(args[0], (list, tuple)):
                        src = self.node_vars[n.args[0]]
                        self.node_vars[n] = f'get<{args[1]}>({src})'
                        return val
                
                no_ops = [
                    'detach',
                    'clone',
                    torch.nn.functional.dropout,
                ]
                if n.target in no_ops:
                    self.alias(n, n.args[0])
                    return val

                out_var = self.get_tmp(n, tuple(val.shape))
                flat_arg_nodes = flatten(n.args)
                flat_arg_vars = [self.deref(n) for n in flat_arg_nodes]
                fargs = ', '.join(flat_arg_vars)

                self.fwds += [f'{out_var} = {fname}({fargs})']
            elif n.op == 'call_module':
                raise ValueError('call_module unsupported')
            elif n.op == 'output':
                self.output_type = val_dtype(val, ref=True)
                self.output_ref = self.nested_refstr(n.args[0])
            else:
                print(n.name, ':', n.op, n.target, n.args, n.kwargs)
                print('->', getattr(val, 'shape', f'{len(val)=}'), len(n.users))

            return val



def codegen(
        model,
        out_file,
        args=[],
        kwargs={},
        tokenizer=None,
        autowrap_functions=[],
        c_prefix='model',
        skip_weights=False,
    ):

    tracer = CustomTracer(autowrap_functions=autowrap_functions)

    graph = tracer.trace(model)

    interp = Interpreter(model, graph=graph)

    out = interp.run(*args, **kwargs)


    if tokenizer is not None:
        n_vocab = tokenizer.get_vocab_size()
        vocab = tokenizer.decode_batch([[i] for i in range(n_vocab)])
        vocab = [bytes(t, 'utf8') for t in vocab]
        token_pack = [struct.pack('B',len(t))+t for t in vocab]
        token_pack = [hex(c) for tok in token_pack for c in tok]
        token_pack = ','.join(token_pack)
        root_tree = {}
        n_trees = 1

        def add_tree(tree, txt, i):
            nonlocal n_trees
            if len(txt)==0:
                tree[''] = i
                return
            if txt[0] not in tree:
                tree[txt[0]] = {}
                n_trees += 1
            add_tree(tree[txt[0]], txt[1:], i)

        for i, txt in enumerate(vocab):
            add_tree(root_tree, txt, i)

    class_name = 'Model'
    blob = []
    ivars = []
    fwds = []

    for name, info in interp.weights.items():
        dtype, val = info
        offset = len(blob)
        blob += val.bfloat16().view(torch.uint16).flatten().tolist()
        ivars += [f'static {dtype} {name} {{ blob+{offset} }}']

    fwds += interp.fwds

    class Writer:
        def __init__(self, f):
            self.f = f
            self.indent = 0

        def __call__(self, s, nl='\n'):
            self.f.write(' ' * self.indent)
            self.f.write(s + nl)
            return self

        def __enter__(self):
            self.__call__('{')
            self.indent += 4
            return self
        
        def __exit__(self, *_):
            self.indent -= 4
            self.__call__('}', '')

    w = Writer(out_file)

    w('#include "torch2cpp/tensor.h"')
    if tokenizer is not None:
        w('#include "torch2cpp/tokenizer.h"')
    w('\n')

    w('namespace {\n')

    if not skip_weights:
        w('// weight initializers"')
        with w(f'static const ml::bfloat16 blob[{len(blob)}] = '):
            w(','.join([hex(x) for x in blob]))
        w(';')

        if tokenizer is not None:
            w(f'uint8_t const g_token_pack[] = {{ {token_pack} }};')
    
    w('// weight tensors')
    for i in ivars:
        w(i + ';')
    w('')

    with w(f'struct {class_name}'):

        w('// inputs')
        for name, dtype in interp.inputs.items():
            w(f'{dtype} {name};')
        w('')
            
        w('// tmp vars')
        for name, info in interp.tmp_vars.items():
            dtype = shape_dtype(info[0])
            w(f'{dtype} {name};')
        w('')

        w(f'{interp.output_type}')
        with w('operator()()'):
            
            w('using std::get;')
            w('using namespace ml;')
                
            for f in fwds:
                w(f + ';')

            w(f'return {interp.output_ref};')

        w('')

    w(';\n\n')

    w(f'ml::rng64 g_rng;')
    w(f'{class_name} g_model;')

    if tokenizer is not None:
        w(f'Tokenizer<{n_vocab}, {n_trees}> g_tokenizer = {{ g_token_pack }};')
    w('\n')
    w('} // namespace\n')

    w(f'''
extern "C" {{
void {c_prefix}_reset()
{{
    std::apply([] (auto &&... x) {{ (x.zero_(), ...); }}, g_model.mem);
}}
int {c_prefix}_step(int prevtok, float temperature)
{{
    g_model.x.ptr()[0] = prevtok;
    auto outs = g_model();
    g_model.mem = std::get<1>(outs);

    auto & logits = std::get<0>(outs);
    return ml::sample_(logits.ptr(), logits.numel(), g_rng, temperature);
}}
''')

    if tokenizer is not None:
        w(f'''
int {c_prefix}_encode(char const* str, int str_len, int * out, int out_len)
{{
    return g_tokenizer.encode(str, str_len, out, out_len);
}}
int {c_prefix}_decode(int const* toks, int toks_len, char * out, int out_len)
{{
    return g_tokenizer.decode(toks, toks_len, out, out_len);
}}
''')

    w('} // extern C\n')