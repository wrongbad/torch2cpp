#include <cstdint>

#include <iostream>

template<int NTOK, int NTREE>
struct Tokenizer
{
    struct Token
    {
        uint8_t length;
        uint8_t str[];
    };
    struct Node
    {
        // next[0] is exit token id
        int token_id = -1;
        int next[256];

        Node() { for(int & i : next) { i = -1; } }
    };
    enum Error
    {
        ERR_OVERFLOW = -1,
        ERR_BUG = -999
    };


    Token const* tokens[NTOK];
    Node tree[NTREE];

    Tokenizer(uint8_t const* token_pack)
    {
        for(Token const* & tok : tokens)
        {
            tok = reinterpret_cast<Token const*>(token_pack);
            token_pack += tok->length + 1;
        }

        // note that null chars are dropped from tokenizations
        int ntree = 1; // tree[0] is root node
        for(int tok_id=0 ; tok_id<NTOK ; tok_id++)
        {
            int node = 0;
            for(int i=0 ; i<tokens[tok_id]->length ; i++)
            {
                uint8_t c = tokens[tok_id]->str[i];
                if(tree[node].next[c] < 0)
                    tree[node].next[c] = ntree++;
                node = tree[node].next[c];
            }
            tree[node].token_id = tok_id;
        }
    }

    // return number of output tokens filled
    // return negative if failed
    int encode(char const* str, int str_len, int * out, int out_len) const
    {
        int out_fill = 0;
        int node = 0;
        int token = -1;
        int token_end = -1;
        for(int i=0 ; i<str_len ; )
        {
            int match = tree[node].token_id;
            if(match >= 0)
            {
                token = match;
                token_end = i;
            }

            uint8_t c = str[i];
            int next = tree[node].next[c];
            if(next >= 0)
            {
                node = next;
                i ++;
            }
            else if(token >= 0)
            {
                if(out_fill >= out_len) { return ERR_OVERFLOW; }
                out[out_fill++] = token;
                i = token_end;
                node = 0;
                token = -1;
                token_end = -1;
            }
            else
            {
                return ERR_BUG;
            }
        }
        int match = tree[node].token_id;
        if(match >= 0)
        {
            if(out_fill >= out_len) { return ERR_OVERFLOW; }
            out[out_fill++] = match;
        }
        return out_fill;
    }

    // return number of output chars filled
    // return negative if failed
    int decode(int const* toks, int toks_len, char * out, int out_len) const
    {
        int out_fill = 0;
        for(int i=0 ; i<toks_len ; i++)
        {
            Token const& tok = *tokens[toks[i]];
            if(out_fill + tok.length > out_len) { return ERR_OVERFLOW; }
            std::copy(tok.str + 0, tok.str + tok.length, out + out_fill);
            out_fill += tok.length;
        }
        return out_fill;
    }
};
