#include <iostream>  

using namespace std;

int function(void) {
	cout << "欧治学好帅" << endl;
	return 0;
}

void registration(int (*rc)(void), int (**func)(void)) {
	*func = rc;
	cout << &rc << endl;
	cout << rc << endl;
}

void fun(int a, int* c){
    cout << a+*c << endl;
}
int (*rc_func)(void);

int main() {

    int (*p)(void)=function; 
    
    cout << &p << endl;
    
    cout << p << endl;

	registration(p, &rc_func);

	rc_func();

    int a=10;
    int c=9;
    fun(10,&c);
	return 0;
}