# music_spotify
## corr_fl.py
The core of the system. Analyse the structure
### init
create objects that contain all necessary data
### Coefficient
| Name | Shape | Definition |
| --- | --- | --- |
| self.inp | (n, m) | input of n dataset with m attributes |
| self.out | (n, k) | output of n dataset with k goals |
| ones | (k, n, 1) | Bias |
| inp | (k, n, m) | copies of input for every k goal|
| out | (k, n, 1) | transformed output so only one k is percieved |
| matrix | (k, n, m+2) | ones + inp + out |
| mat | (k, m+1, m+2) | describe XT * (XM = Y) as XT(XY) |
| A | (k, m+1, m+1) | XTX |
| b | (k, m+1, 1) | XTY |
| LAMBDA | number | L2 Regulation |
| LAMBDA | (k, m+1, m+1) | Modified Regulation |
| norm | boolean | standarise the solution