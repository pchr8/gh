import random;
import numpy as n;
import sys;
import os;
import time;

###
# SETTINGS
###

#Width, height
w, h = 4, 4;
w, h = 43, 43;
w, h = 13, 13;

###
# OUTPUT
###

# Print Glyph# {{
def prGl(tch):
    for i in range(w):
        for j in range(h):
            if (tch[i][j]):
                sys.stdout.write("█");
            else:
                sys.stdout.write("░");
        print('');
    print('');
# }}
# {{ Print  DNA
def prDNA(d):
    for i in range(len(d)):
        if(d[i]):
            sys.stdout.write('1');
        else:
            sys.stdout.write('0');
    print();
# }}

###
# CONVERSIONS
###

# {{ DNA to Glyph
def DNAtoGl(dna):
    global w;
    global h;
    ch = [[bool(random.getrandbits(1)) for y in range(w)] for x in range(h)]
    for i in range(h):
        for j in range(w):
            ch[i][j]=dna[h*i+j];
    return ch;
# }}

###
# GENERATION
###

# {{ Generate random DNA
def DNAGen():
    global w; global h;
    d = [bool(random.getrandbits(1)) for i in range(h*w)];
    return d;
# }}
# {{ Pretty DNA gen
def prettyDNAgen():
    global w; global h;
    d = n.zeros((h*w))
    
    x, y=0, 1; # starting point
    prev=[x, y];
    d[h*(y)+x]=1;
    while ((abs(x)<w-1)): #both -1 because max is (6, 6);
    #while ((x<w-1) or (y<h-1)): #both -1 because max is (6, 6);
        #os.system('clear');  
        #print("<", x, y);
        while ((x==prev[0]) and (y==prev[1])):
            '''if ((x>0) and (x<w-2)):
                xr=random.randint(0,2);
                x+=xr;
                x=x-1;
            else:
                xr=random.randint(0,1);
                x+=xr;
            #print(">", x, y);'''
            if (y>h-2):
                yr=random.randint(0,1);
                y-=yr;
            elif ((y>0)):
                yr=random.randint(0,2);
                y+=yr;
                y=y-1;
            else:
                yr=random.randint(0,1);
                y+=yr;
                #print(">>>", x, y);

            if (x>w-2):
                xr=random.randint(0,1);
                x-=xr;
            elif ((x>0)):
                xr=random.randint(0,2);
                x+=xr;
                x=x-1;
            else:
                xr=random.randint(0,1);
                x+=xr;
                #print(">>>", x, y);
        prev[0]=x;
        prev[1]=y;

        #print(">", x, y);
        #print(w,"*",y-1,"+",x,"-1=",w*y+x-1);
        #print(h,"*",y,"+",x,"=",h*y+x);
        d[h*(y)+x]=1;
        #prGl(DNAtoGl(d));
        #time.sleep(0.05);

    return d;
# }}


###
# START PROGRAM
###
start_time = time.time()

random.seed();

'''
print("Normal:");
d = DNAGen();
prDNA(d);
g = DNAtoGl(d);
prGl(g);
'''

#print("Pretty:");
p = prettyDNAgen();
#prDNA(p);
for i in range (1000):  
    pg = DNAtoGl(p);
#prGl(pg);
print("--- %s seconds ---" % (time.time() - start_time))
