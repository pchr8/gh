# General
# I need a way to quantify how "far away from each other" the symbols are.
# https://en.wikipedia.org/wiki/Dynamic_time_warping the possibilities I seem 
# https://nbviewer.jupyter.org/github/pierre-rouanet/dtw/blob/master/simple%20example.ipynb excellent on the same topic
# to find are only for graphs, but I may have loops
# Calculating the distance between all sets of points? 

# I might be looking for something like this:
# https://stackoverflow.com/questions/1819124/image-comparison-algorithm
# Basically create some sort of historgram for X and Y
# and then do a dynamic time warp thing.


import random;# {{
from math import sin, cos, radians
import math;

import sys;
import os;
import time;
import numpy as n; # currently used only in n.zeroes# }}
import operator; # https://stackoverflow.com/questions/4010322/sort-a-list-of-class-instances-python
import matplotlib.pyplot as plot
from matplotlib import pyplot

import cv2; #http://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist#comparehist



###
# SETTINGS
###
# {{
#Width, height
w, h = 4, 4;
w, h = 10, 10;
numberToDraw = 1000;
# }}

class Glyph:
    global w; global h;
    points = ();
    dna = n.zeros((h*w))
    rating = 1000;
    
    def getRating(self):
        # Optimal number of points 
        points=3;

        if (self.points[len(self.points)-1][0]<w-1):
            rating-=200;

        self.rating -= abs(points-len(self.points));
#TODO. I want to take the smallest,  so it works for both 1/4 and 3/4 height
        self.rating-=abs(
                round(int(
                    min(
                    self.points[len(self.points)-1][1]-
                        h*(3/4), 
                    self.points[len(self.points)-1][1]-
                        h/4)
                        
                    ))
                );
        return int(self.rating);

    def __init__(self, data):
        #print("Hello there!");
        '''
        self.dna = n.zeros((h*w))
        self.points = [(0,0)];
        self.wh = (w, h);
        '''
        self.points=data[1];
        self.dna=data[0];
        self.rating=self.getRating();
    def prDNA(self):
        prDNA(self.dna);
    def prPoints(self):
        print("My points are:");
        print(self.points);
    def prGl(self):
        prGl(DNAtoGl(self.dna));
    def prHistGl(self):
        hists=self.getHistogram();
        tch=DNAtoGl(self.dna);

        for i in range(w):
            sys.stdout.write(str(int(hists[1][i])));
        print('');
        for i in range(w):
            for j in range(h):
                if (tch[i][j]):
                    sys.stdout.write("█");
                else:
                    sys.stdout.write("░");
            print(int(hists[0][i]));
        print('');

    def startEnd(self):
        ex = self.points[len(self.points)-1][0];
        ey = self.points[len(self.points)-1][1];
        sx = self.points[0][0];
        sy = self.points[0][1];
        return(ex, ey, sx, sy);

    def __repr__(self):
        return repr((self.points, self.dna, self.rating));

    def getHistogram(self):
        tch = DNAtoGl(self.dna);

        histx = [];
        histy = [];

        row = 0;
        for i in range(w):
            for j in range(h):
                row+=tch[i][j];
            histx.append(row);
            row= 0;
        for i in range(h):
            for j in range(w):
                row+=tch[j][i];
            histy.append(row);
            row= 0;
        return(histx, histy);






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
# VARIOUS
###

def findpx(x0, y0, x1, y1):# {{
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    pixels = []
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x0
        x0 = y0
        y0 = t

        t = x1
        x1 = y1
        y1 = t

    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t

        t = y0
        y0 = y1
        y1 = t

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = round(x0)
    y_end = y0 + (gradient * (x_end - x0))
    xpxl0 = x_end
    ypxl0 = round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl1 = x_end
    ypxl1 = round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in range(xpxl0 + 1, xpxl1):
        if steep:
            pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pixels# }}

def get_line(start, end):# {{
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points# }}

def putPoint(x, y, dna):# {{
    global w; global h;

    if (x>w-1):
        x=w-1;
    if (y>h-1):
        y=h-1;
    dna[h*(y)+x]=1;
    return dna;# }}

###
# CONVERSIONS AND COMPARISONS
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

def compare(g1, g2):
    x1 = n.asarray(g1.getHistogram()[1]);
    x2 = n.asarray(g2.getHistogram()[1]);
    y1 = n.asarray(g1.getHistogram()[0]);
    y2 = n.asarray(g2.getHistogram()[0]);

    diffx = cv2.compareHist(n.float32(x1), n.float32(x2), 3);
    diffy = cv2.compareHist(n.float32(y1), n.float32(x2), 3);
    #print(diffx, "+", diffy, "=", diffx+diffy);
    return(diffx+diffy);

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
    #d[h*(y)+x]=1;
    putPoint(x, y, d);
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
        putPoint(x, y, d);
        #prGl(DNAtoGl(d));
        #time.sleep(0.05);

    return d;
# }}

# {{ Pretty DNA gen 2
def pDNAg2():
    global w; global h;
    d = n.zeros((h*w))
    
    x, y=0, round(int(h/4)); # starting point
    fx, fy=x, y; # original points;
    putPoint(x, y, d);

    # From here, we generate a radius and a point on that radius
    # Loop start

#    for i in range(8):
    while x<w:

        radius=int(round(random.randint(1, int(min(w, h)/2))));
        direction=random.randint(1, 360);
        xx=x+(int(round(sin(radians(direction))*radius)));
        yx=y+(int(round(cos(radians(direction))*radius)));

        #while not (((xx>0) and (xx<w-1)) and ((yx>0) and (yx<h-1))):
        while not (((xx>0)) and ((yx>0) and (yx<h-1))):
            #print("Radius: ", radius, "; Direction: ", direction);
            #print("x: ", xx, "; y: ", yx);
            radius=int(round(random.randint(1, int(min(w, h)/2))));
            direction=random.randint(1, 360);
            xx=x+(int(round(sin(radians(direction))*radius)));
            yx=y+(int(round(cos(radians(direction))*radius)));

        x=xx; 
        y=yx;
        #d[h*(y)+x]=1;
        putPoint(x, y, d);
        #print("Radius: ", radius, "; Direction: ", direction);
        #print("x: ", x, "; y: ", y);
        #px = findpx(fx, fy, x, y);
        px = get_line((fx, fy), (x, y));
        for (dx, dy) in px:
            #print (x, y);
            #d[h*(dy)+dx]=1;
            putPoint(dx, dy, d);
        fx=x;
        fy=y;
    return d;
'''
    while ((abs(x)<w-1)): #both -1 because max is (6, 6);
    #while ((x<w-1) or (y<h-1)): #both -1 because max is (6, 6);
        #os.system('clear');  
        #print("<", x, y);
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

        #print(">", x, y);
        #print(w,"*",y-1,"+",x,"-1=",w*y+x-1);
        #print(h,"*",y,"+",x,"=",h*y+x);
        d[h*(y)+x]=1;
        #prGl(DNAtoGl(d));
        #time.sleep(0.05);
'''
# }}

def class_pDNAg2():
    global w; global h;
    d = n.zeros((h*w))
    
    points = ([(0, round(int(h/4)))]); # starting point

    x, y=0, round(int(h/4)); # starting point
    fx, fy=x, y; # original points;
    d[h*(y)+x]=1;
    putPoint(x, y, d);
    

    # From here, we generate a radius and a point on that radius
    # Loop start

#    for i in range(8):
    while x<w:

        radius=int(round(random.randint(1, int(min(w, h)/2))));
        direction=random.randint(1, 360);
        xx=x+(int(round(sin(radians(direction))*radius)));
        yx=y+(int(round(cos(radians(direction))*radius)));

        #while not (((xx>0) and (xx<w-1)) and ((yx>0) and (yx<h-1))):
        while not (((xx>0)) and ((yx>0) and (yx<h-1))):
            #print("Radius: ", radius, "; Direction: ", direction);
            #print("x: ", xx, "; y: ", yx);
            radius=int(round(random.randint(1, int(min(w, h)/2))));
            direction=random.randint(1, 360);
            xx=x+(int(round(sin(radians(direction))*radius)));
            yx=y+(int(round(cos(radians(direction))*radius)));

        x=xx; 
        y=yx;
        #d[h*(y)+x]=1;
        putPoint(x, y, d);
        if (x>w-1):
            points.append((w, y));
        if (y>h-1):
            points.append((x, h));
        else:
            points.append((x, y));
        #print("Radius: ", radius, "; Direction: ", direction);
        #print("x: ", x, "; y: ", y);
        #px = findpx(fx, fy, x, y);
        px = get_line((fx, fy), (x, y));
        for (dx, dy) in px:
            #print (x, y);
            #d[h*(dy)+dx]=1;
            putPoint(dx, dy, d);
        fx=x;
        fy=y;
    #return d;
    return (d, points);

###
# START PROGRAM
###

random.seed();

'''
start_time=time.time()
glyphs=[];
numberToDraw = 5000;
for i in range (numberToDraw):  
    bx = Glyph(class_pDNAg2());
    glyphs.append(bx);

#now we have glyphs[] with 1000 different glyphs.
middle=time.time();

sgl = sorted(glyphs, key=lambda glyph: glyph.rating) ;
#for i in n.arange(0, len(sgl), 50):
for i in n.arange(len(sgl)-round(int(len(sgl)/10)), len(sgl), 1):
    print(sgl[i].getRating(), ":");
    print(sgl[i].startEnd());
    sgl[i].prGl();

end=time.time();
print("Middle: %s seconds ---" % (middle-start_time));
print("Drawing %s seconds ---" % (end-start_time));


'''

'''
start_time=time.time()

bx = Glyph(class_pDNAg2());
bx.prHistGl();
ax = Glyph(class_pDNAg2());
ax.prHistGl();

print(compare(ax, bx));

end=time.time();
print("Time taken: %s seconds ---" % (end-start_time));
'''


glyphs=[];
for i in range (numberToDraw):  
    bx = Glyph(class_pDNAg2());
    glyphs.append(bx);

sgl = sorted(glyphs, key=lambda glyph: glyph.rating) ;


#https://stackoverflow.com/questions/646644/how-to-get-last-items-of-a-list-in-python
good = sgl[-round(int(len(sgl)/15)):];

ds = n.empty(shape=(len(good), len(good)));

for i in range(len(good)):
    for j in range(len(good)):
        ds[i][j]=compare(good[i], good[j]);

print(ds);

'''
for i in range(len(good)):
    print("==================================================");
    good[i].prHistGl();
    print("Max:", ds[i][n.argmax(ds[i])]);
    good[n.argmax(ds[i])].prHistGl();
    print("Min:", ds[i][n.argmin(ds[i])]);
    good[n.argmin(ds[i])].prHistGl();
'''


# https://stackoverflow.com/questions/26603747/get-the-indices-of-n-highest-values-in-an-ndarray !!!!

#LEAST ALIKE
print("==================================================");
indices =  n.argpartition(ds.flatten(), -2)[-2:]; # top 3
print(indices);
print(n.vstack(n.unravel_index(indices, ds.shape)).T);
print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[0][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[0][1]]);
print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[1][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[1][1]]);

print("least alike:");
good[n.vstack(n.unravel_index(indices, ds.shape)).T[1][0]].prHistGl();
good[n.vstack(n.unravel_index(indices, ds.shape)).T[1][1]].prHistGl();
print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[1][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[1][1]]);


good[n.vstack(n.unravel_index(indices, ds.shape)).T[0][0]].prHistGl();
good[n.vstack(n.unravel_index(indices, ds.shape)).T[0][1]].prHistGl();
print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[0][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[0][1]]);
'''
#MOST ALIKE
print("==================================================");
indices =  n.argpartition(ds.flatten(), 2)[:2]; # top 3
print(indices);
print(n.vstack(n.unravel_index(indices, ds.shape)).T);
print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[0][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[0][1]]);
print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[1][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[1][1]]);

print("most alike:");
good[n.vstack(n.unravel_index(indices, ds.shape)).T[1][0]].prHistGl();
good[n.vstack(n.unravel_index(indices, ds.shape)).T[1][1]].prHistGl();
print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[1][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[1][1]]);

good[n.vstack(n.unravel_index(indices, ds.shape)).T[0][0]].prHistGl();
good[n.vstack(n.unravel_index(indices, ds.shape)).T[0][1]].prHistGl();
print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[0][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[0][1]]);
'''



'''
for i in range(len(good)):
    indices =  n.argpartition(ds.flatten(), -1)[-1:]; # top 2
    print(indices);
    print(n.vstack(n.unravel_index(indices, ds.shape)).T);
    print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[0][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[0][1]]);
    print(ds[n.vstack(n.unravel_index(indices, ds.shape)).T[1][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[1][1]]);

    print("==================================================");
    good[i].prHistGl();
    print("Max:",ds[n.vstack(n.unravel_index(indices, ds.shape)).T[0][0]][n.vstack(n.unravel_index(indices, ds.shape)).T[0][1]] );

#    good[n.argmax(ds[i])].prHistGl();
'''

# Now the idea is to find N elements such that their sum is maximum
# Select combination of elements from array such that their sum is maximized

# Yes, last thing for today:

# This could work as a genetic algorithm when connected to svgs. 

# Bezier curves python! N points, etc etc.
