pragma circom 2.0.0;

template Test1 () {  
   signal input a;  
   signal input b;  
   signal output c;  

   c <== a * b;  
}

component main = Test1();