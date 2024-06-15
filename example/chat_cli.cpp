#include <string>
#include <iostream>

extern "C" {
void model_reset();
int model_step(int prevtok, float temperature);
int model_encode(char const* str, int str_len, int * out, int out_len);
int model_decode(int const* toks, int toks_len, char * out, int out_len);
} // extern C

int main(int argc, char ** argv)
{
    std::string prompt;
    constexpr int max_tokens = 256;
    int toks[max_tokens];
    char decode[max_tokens];
    float temperature = 1;
    while(true)
    {
        std::getline(std::cin, prompt);
        // prompt += "\n";

        int n_tok = model_encode(prompt.c_str(), prompt.size(), toks, max_tokens);

        std::cout << n_tok << std::endl;
        for(int i=0 ; i<n_tok ; i++)
            std::cout << toks[i] << ", ";
        std::cout << std::endl;

        for(int i=0 ; i<n_tok-1 ; i++)
        {
            model_step(toks[i], 0);
        }
        for(int i=n_tok-1 ; i<max_tokens ; i++)
        {
            toks[i+1] = model_step(toks[i], temperature);
            int strlen = model_decode(toks+i+1, 1, decode, max_tokens);
            std::cout << std::string(decode, strlen);
            bool exit = false;
            for(int j=0 ; j<strlen ; j++)
            {
                exit |= (decode[j]=='\n');
            }
            if(exit) { break; }
        }
        std::cout << std::endl;
    }
}