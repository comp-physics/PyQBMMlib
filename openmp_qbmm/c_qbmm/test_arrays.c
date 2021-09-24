#include <stdio.h> 
  
void compare(int* add_great, int* add_small) 
{ 
    *add_great[0] = 5;
    *add_small[0] = 4; 

    *add_great[1] = 7;
    *add_small[1] = 9; 
} 
  
// Driver code 
int main() 
{ 
    int great[2], small[2], x, y; 
  
    compare(&great, &small); 
    /* printf("The greater number is %d and the smaller number is %d\n", */ 
           /* great[0], small[0]); */ 
  
    return 0; 
} 
