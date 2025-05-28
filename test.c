#include <stdint.h>

extern void print_string(char* c, int m);

char* str1="the 1st one\n";
char* str2="the 2st one\n";



int bar_func(int a,int b){
    if(a<b){
        print_string(str1,13);

    }
    else{
        print_string(str2,13);

    }

    return 0;
}

int func(int* a,int* b){
    if(*a<*b){
        print_string(str1,13);

    }
    else{
        print_string(str2,13);

    }

    return 0;
}
