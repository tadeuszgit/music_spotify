# music_spotify
## corr_fl.py
The core of the system. Analyse the structure
### init
create objects that contain all necessary data
### Coefficient
Uses the self.inp and self.out from the object. They are defined in other function. The self.inp is matrix of output and self.inp is matrix of input. The amount of row needs to MATCH otherwise the function crash because each row represent different data set that contains multiple input and output
self.inp (n,m)
self.out (n,k)
ones (k,n,1)
inp (k,n,m)
out (k,n,1)
matrix (k,n,m+2) aka our whole dataset
mat (k,m+1,m+2) it is analog to XM = Y -> XTXM = XTY
So XTX is A (k,m+1,m+1) without the last Y-column and YTX is b (k,m+1,1) just the column with Y

Additionally we can add L2 Regulation that use standard devation of the input or predefined Regulation with the shape of (k,m+1,m+1)
In order to get standarise solution we subtract the mean and divide by the standardevation