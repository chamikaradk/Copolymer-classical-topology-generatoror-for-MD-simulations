# Copolymer-classical-topology-generator-
Generates GROMACS topology and parameter files for mixed random copolymer distributions. copolymers can contain up to two types of monomers (tested for N2200 alkylated and glycolated polymer chains, modeling P10 to P90 chains). Ex: To get a 10mer chain with alkyl side chain groups attached to the 7th monomer unit, simply set the array input to  [1,1,1,1,1,1,0,1,1,1]. By using the dist.ipynb code, you can automatically generate polymer distributions of P90 chains.  

1. Input the information about the two polymers inlcuding DFT calculated potentials
   example inputs are provided, which is the N2200 monomer and it's glycolated chain versions
3. Define the polymer experimental information (Mn, Mw, PDI)
4. Generate polymer distributions

Detailed info with diagrams provided in the jupyter notebook 
![image](https://github.com/user-attachments/assets/91fcdae2-6ff3-41cf-a2e2-92b5c36f1b28)

