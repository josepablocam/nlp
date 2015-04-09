//Used to clean up data as collected for trainining
f:first .Q.opt[.z.x]`file; //file
if[0=count f; show "need an input file"; exit 1];
if["1"~first first system"test -f ",f,";echo $?"; show "Input file not found"; exit 1]; 
nf:"."sv @["." vs f;0;,[;"_mod"]]; //new file for output
if["0"~first first system"test -f ",nf,";echo $?"; show "Output file already exists"; exit 1];
t:trim each read0 hsym `$f; //readfile
p:".!?,\";:()_-"; //punctuation to replace with spaces
s:trim (ssr[;2#" ";" "]/)@ssr/[;p;" ",/:p,\:" "]@ //add spaces around punctuation and then remove extra spaces
w:where 0<>count each t; //index of non-empty lines
l:"/O"; //default tag
m:{" "sv(" "vs trim x),\:l} //tag function
(hsym`$nf) 0:@[t;w;m s@] //write out modified file
exit 0
