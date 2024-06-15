# torch2cpp

Status: WIP

Some features are supported, but most are not yet.

Models are traced with torch.fx to extract the flattened AST of low level function calls.

The AST graph is then traversed, generating C++ code in order.

Temporary buffer shapes and ref-counts are tracked to enable compile-time scheduled memory re-use (~10x buffer reduction in common cases).

Weights are stored in the binary as bfloat16, and unpacked to float32 at runtime. (Would like to investigate more options here for both storage and inference)

The bundled tensor math lib uses compile-time shapes and in-place storage, so there is no dynamic memory allocation at all.

Headers are installed with python packge, can be found with `python -m torch2cpp.includes`