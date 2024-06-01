#include <memory>
#include <cassert>
#include <cmath>
#include <tuple>

namespace ml {

struct bfloat16 { uint16_t data; };


// only a truly insane person would write a tensor math library
// using c++ template metaprogramming


template<int... dims>
struct tensor;


template<int dim0, int... dims>
struct tensor<dim0, dims...>
{
    using SubT = tensor<dims...>;

    template<int D>
    using StackT = tensor<D, dim0, dims...>;

    SubT data[dim0];

    tensor() {}
    tensor(bfloat16 const* src)
    {
        // little endian only!!!
        uint16_t * bf = reinterpret_cast<uint16_t*>(ptr());
        for(int i=0 ; i<numel() ; i++)
        {
            // bf[i * 2 + 0] = 1<<15;
            bf[i * 2 + 1] = src[i].data;
        }
    }
    static constexpr int numel()
    {
        return dim0 * SubT::numel();
    }
    SubT & operator[](int i) { return data[i]; }
    SubT const& operator[](int i) const { return data[i]; }
    float item() const { return data[0].item(); }
    float * ptr() { return reinterpret_cast<float*>(&data); }
    float const* ptr() const { return reinterpret_cast<float const*>(&data); }

    tensor & zero_()
    {
        for(int i=0 ; i<numel() ; i++)
            ptr()[i] = 0;
        return *this;
    }
};

template<>
struct tensor<>
{
    float f = 0;

    tensor() {}
    static constexpr int numel() { return 1; }
    float item() const { return f; }
    operator float() const { return f; }
    tensor<> & operator=(float x) { f=x; return *this; }
    tensor<> & operator+=(float x) { f+=x; return *this; }

    tensor & zero_()
    {
        f = 0;
        return *this;
    }
};




template<int... Sx>
struct slice {};

template<int... dims>
auto getitem(tensor<dims...> & x)
{
    return x;
}

template<int dim0, int... dims, int... Sx, class... Slices>
auto getitem(tensor<dim0, dims...> & x, slice<Sx...> s, Slices... sss)
{
    static_assert(sizeof...(Sx) == 0, "not implemented yet");
    
    using OutT = typename decltype(getitem(x[0], sss...))::template StackT<dim0>;
    OutT out;
    for(int i=0 ; i<dim0 ; i++)
    {
        out[i] = getitem(x[i], sss...);
    }
    return out;
}

template<int dim0, int... dims, class... Slices>
auto getitem(tensor<dim0, dims...> & x, int i, Slices... sss)
{
    if(i < 0) i += dim0;
    return getitem(x[i], sss...);
}




template<int H, int W>
tensor<W> embedding(tensor<> const& x, tensor<H, W> const& m)
{
    return m[int(x.item())];
}

template<int dim0, int... dims, int H, int W>
auto embedding(tensor<dim0, dims...> const& x, tensor<H, W> const& m)
{
    using OutT = typename decltype(embedding(x[0], m))::template StackT<dim0>;
    OutT out;
    for(int i=0 ; i<dim0 ; i++)
    {
        out[i] = embedding(x[i], m);
    }
    return out;
}



template<int... dims>
tensor<dims...> add(tensor<dims...> const& x, tensor<dims...> const& y)
{
    tensor<dims...> out;
    for(int i=0 ; i<out.numel() ; i++)
        out.ptr()[i] = x.ptr()[i] + y.ptr()[i];
    return out;
}

template<int... dims>
tensor<dims...> mul(tensor<dims...> const& x, tensor<dims...> const& y)
{
    tensor<dims...> out;
    for(int i=0 ; i<out.numel() ; i++)
        out.ptr()[i] = x.ptr()[i] * y.ptr()[i];
    return out;
}


// https://github.com/ekmett/approximate/blob/master/cbits/fast.c
float fast_exp(float x)
{
    union { float f; int32_t i; } p, n;
    p.i = 1056478197 + int32_t(6051102 * x); // exp(x/2)
    n.i = 1056478197 - int32_t(6051102 * x); // exp(-x/2)
    return p.f / n.f;
}

float fast_sigmoid(float x)
{
    union { float f; int32_t i; } p, n;
    p.i = 1056478197 + int32_t(6051102 * x); // exp(x/2)
    n.i = 1056478197 - int32_t(6051102 * x); // exp(-x/2)
    return p.f / (p.f + n.f);
}

float fast_tanh(float x)
{
    union { float f; int32_t i; } p, n;
    p.i = 1064866805 + int32_t(12102203 * x); // exp(x)
    n.i = 1064866805 - int32_t(12102203 * x); // exp(-x)
    return (p.f - n.f) / (p.f + n.f);
}

float softsign(float x)
{
    return x / (std::abs(x) + 1); 
}


template<int... dims>
tensor<dims...> sigmoid(tensor<dims...> const& x)
{
    tensor<dims...> out;
    for(int i=0 ; i<out.numel() ; i++)
        out.ptr()[i] = fast_sigmoid(x.ptr()[i]);
    return out;
}

template<int... dims>
tensor<dims...> softsign(tensor<dims...> const& x)
{
    tensor<dims...> out;
    for(int i=0 ; i<out.numel() ; i++)
        out.ptr()[i] = softsign(x.ptr()[i]);
    return out;
}


template<int W>
tensor<W> rms_norm(tensor<W> const& x, tensor<W> const& m, float eps)
{
    float norm = 0;
    for(int i=0 ; i<W ; i++)
        norm += x[i] * x[i];
    norm = norm / W + eps;
    norm = 1 / std::sqrt(norm);

    tensor<W> out;
    for(int i=0 ; i<W ; i++)
        out[i] = norm * m[i] * x[i];
    return out;
}

template<int dim0, int... dims, int W>
tensor<dim0, dims...> rms_norm(tensor<dim0, dims...> const& x, tensor<W> const& m, float eps)
{
    tensor<dim0, dims...> out;
    for(int i=0 ; i<dim0 ; i++)
    {
        out[i] = rms_norm(x[i], m, eps);
    }
    return out;
}




template<int H, int W>
tensor<H> linear(tensor<W> const& x, tensor<H, W> const& m, nullptr_t const&)
{
    // static constexpr int T = 16;
    // static_assert(W % T == 0, "");
    tensor<H> out;
    // TODO optimize
    for(int i = 0 ; i < H ; i++)
    {
        out[i] = 0;
        for(int j = 0 ; j < W ; j++)
        {
            out[i] += x[j].f * m[i][j].f;
        }
    }
    return out;
}
template<int H, int W>
tensor<H> linear(tensor<W> const& x, tensor<H, W> const& m, tensor<H> const& b)
{
    tensor<H> out;
    // TODO optimize
    for(int i = 0 ; i < H ; i++)
    {
        out[i] = b[i];
        for(int j = 0 ; j < W ; j++)
        {
            out[i] += x[j] * m[i][j];
        }
    }
    return out;
}

template<int dim0, int... dims, int H, int W, class Bias>
auto linear(tensor<dim0, dims...> const& x, tensor<H, W> const& m, Bias const& b)
{
    using OutT = typename decltype(linear(x[0], m, b))::template StackT<dim0>;
    OutT out;
    for(int i=0 ; i<dim0 ; i++)
    {
        out[i] = linear(x[i], m, b);
    }
    return out;
}

template<int T, int W>
tensor<T, W> sqrll_kernel(
    tensor<T, W> const& x,
    tensor<T, W> const& r,
    tensor<W> const& p)
{
    tensor<T, W> out;

    for(int j=0 ; j<W; j++)
        out[0][j] = x[0][j].f + r[0][j].f * p[j].f;
    for(int t=1 ; t<T ; t++)
        for(int j=0 ; j<W; j++)
            out[t][j] = x[t][j].f + r[t][j].f * out[t-1][j].f;
    
    return out;
}

// compiler has trouble deducing dim0 if we use it for r and p
template<int dim0, int... xdims, int... rdims, int... pdims>
tensor<dim0, xdims...> sqrll_kernel(
    tensor<dim0, xdims...> const& x,
    tensor<rdims...> const& r,
    tensor<pdims...> const& p)
{
    tensor<dim0, xdims...> out;
    for(int i=0 ; i<dim0 ; i++)
    {
        out[i] = sqrll_kernel(x[i], r[i], p[i]);
    }
    return out;
}



struct rng64
{
    uint64_t x = 1234567890;

    float operator()()
    {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return (x % int(1e6)) / 1e6;
    }
};

int sample_(float * x, int n, ml::rng64 & rng, float temperature=1)
{
    if(temperature < 0.01)
    {
        int out = 0;
        for(int i=0 ; i<n ; i++)
            if(x[i] > x[out]) out=i;
        return out;
    }

    float sum_exp = 0;
    for(int i=0 ; i<n ; i++)
    {
        x[i] = (x[i] > -40) ? ml::fast_exp(x[i] / temperature) : 0;
        sum_exp += x[i];
    }
    float thresh = rng() * sum_exp;
    float cumprob = 0;
    for(int i=0 ; i<n ; i++)
    {
        cumprob += x[i];
        if(cumprob > thresh) { return i; }
    }
    return n-1;
}

} // namespace ml
