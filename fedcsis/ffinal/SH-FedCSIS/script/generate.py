# generation of glyphs for shorthand system.

from math import sin, cos, radians, hypot, atan2, degrees, pi
import sys;
import os;
import time;
import numpy as n;
import operator; # https://stackoverflow.com/questions/4010322/sort-a-list-of-class-instances-python
import matplotlib.pyplot as plt
import seaborn
import random;
import cv2; #http://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist#comparehist
import bezier

random.seed();

###
# {{ # SETTINGS
###

# How many are drawn
numberToDraw = 120;

# Max number of control points in the Bezier curves
maxControlPoints = 40; 

# Which part of the elementns will be considered "good". 1/thold is picked.
thold=7;

# }}

class Glyph:
    nOfPoints=0;
    points = ();
    rating = 1000;

    type = [0, 0]; # Where is the beginning and the end

    def __init__(self, data):
        self.points=data;
        self.vector = bezier.Curve.from_nodes(data)
        self.length=self.vector.length;
        self.fauxHistogram=self.getSumOfTurns(self.points)[1];
        self.sumOfTurns=self.getSumOfTurns(self.points)[0];
        self.rating=self.getRating();
        self.countType();

    def getRating(self):
        rating=1000;

#~Uglovatost', the less the better.
        howManyAngles=self.sumOfTurns/self.length;
        self.rating=1000-howManyAngles;
        return int(self.rating);

    def printPoints(self):
        return self.points;

    def getFauxHistogram(self):
        return self.fauxHistogram;

# Find type of connection at the beginning and at the end

    def countType(self):
        data = self.printPoints();

        if round(data[0][1])==5:
            self.type[0]=1;
        elif round(data[0][1])==2:
            self.type[0]=0;
        elif round(data[0][1])==8:
            self.type[0]=2;
        else:
            print("WRX");

        if round(data[len(data)-1][1])==5:
            self.type[1]=1;
        elif round(data[len(data)-1][1])==2:
            self.type[1]=0;
        elif round(data[len(data)-1][1])==8:
            self.type[1]=2;

    def getType(self):
        return self.type;

    def getLength(self):
        return self.length;
    
    def getSumOfTurns(self, points):
        sumAngles=0;
        allAngles=[];
        for i in range(1, len(points)):
            currAngle = angle(
                    points[i][0], 
                    points[i-1][0], 
                    points[i][1], 
                    points[i-1][1]
                    );
            sumAngles+=abs(currAngle);
            allAngles=n.append(allAngles, currAngle);
        return(sumAngles, allAngles);

    def graphMe(self, what, where):

        ax = self.vector.plot(num_pts=256)
        plt.axis([0, 10, 0, 10]);
        #Set a bit of nice graphics, esp. show connection points:
        ym = plt.plot(0.1, 10.0/6, 'bs', 0.1, 10.0/6+(10.0/6)*2, 'bs',  0.1, 10.0/6+(10.0/6)*4, 'bs');
        yy=[10.0/6.0, 10.0/6.0];
        for a in range(0, 6, 2):
            lm = plt.plot([0, 10], [y+y*a for y in yy], linewidth=0.3, color='r');

        # Show some values:
        plt.text(2, 8, "length: "+str(self.length));
        plt.text(2, 7, "sum of angles: "+str(self.sumOfTurns));
        plt.text(2, 6, "Rating: "+str(self.getRating()));
        plt.text(2, 6.5, "Control points: "+str(len(self.points)));
        self.countType();
        plt.text(2, 5.5, "Type of connection, start and end: "+
                str(self.getType())
                );

        if what=="out":
            print("Showing symbol:");
            plt.show();
        else:
            filename = "imgs/"+where+".svg"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            print("Saving symbol as ", where);
            plt.savefig("imgs/"+where+".svg");

        #To control the warning about too many figures open and to save CPU:
        plt.close("all");


    # Make bezier curve one dimension bigger, to make comparing histograms easier.
    def elevate(self):
        self.vector=self.vector.elevate();
        self.points=self.vector.nodes;
        self.fauxHistogram=self.getSumOfTurns(self.points)[1];
        self.rating=self.getRating();

    # For sorting
    def __repr__(self):
        return repr((self.points,  self.rating));
        

#####
# HELPER FUNCTIONS
#####

#Angle between two lines
def angle(x1, x2, y1, y2):
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return degs;

#Compare two symbols
def compare(g, k):
    print("Comparing "+str(g.getRating())+" and "+str(k.getRating()));

#Histograms:

    fg=g.getFauxHistogram();
    fk=k.getFauxHistogram();
    diffx = cv2.compareHist(n.float32(fg), n.float32(fk), 1);

    print("Histogram difference: "+str(g.getRating())+" and "+str(k.getRating())+"="+str(diffx));

#Rating

    frg=g.getRating();
    frk=k.getRating();
    r=(frg+frk)/4;
    r=0;

    print("Ratings:"+str(r));

    final=diffx+r*10;  #Tweak

    print("Final difference between glyphs:"+str(final));

    return(final);

#########
# GENERATE 
#########

#Generates array of points and control points.

def GeneratePoints(start, end, nOfPoints):

    # First point
    rand = n.array( [[0, (10.0/6)*2*start+10.0/6]])

    #Control points;
    for i in range (nOfPoints):  
        rpoints = [[random.randint(0, 10), random.randint(0, 10)]] ;
        rand=n.append(rand, rpoints, axis=0);

    #Last point:
    rand=n.append(rand,[[10,  (10.0/6)*2*end+10.0/6]], axis=0);

    return rand;

####################################################
# START PROGRAM
####################################################

glyphs=[];

#############
## GENERATE GLYPHS ##
#############

for i in range (numberToDraw):  
    mu, sigma=3, int(maxControlPoints/4); #Tweak
    '''s=abs(n.random.normal(mu, sigma, 1000));
    t3=plt.hist(s, 30, normed=True);
    plt.show();
    '''

    nOfPoints    = int(abs(n.random.normal(mu, sigma)))
    #nOfPoints    = random.randint(0, maxControlPoints-2);
    start=random.randint(0, 2);
    end=random.randint(0, 2);

    #print(start, end, nOfPoints);
    bx = Glyph(GeneratePoints(start, end, nOfPoints));
    glyphs.append(bx);
    bx.graphMe("save", "all/"+str(i)+":"+str(bx.getRating())+":"+str(bx.getLength()));


#############
## SORT GLYPHS ##
#############

##  sgl = sorted(glyphs, key=lambda glyph: glyph.rating) ;

#############
## GET BEST GLYPHS ##
#############


##Probab. distributions

##  howmany = int(len(sgl)/thold);
##  center = len(sgl)-int(2.5*thold) #mu #Tweak
##  sigma = int(len(sgl)/(2*thold)); #Tweak

##  print(
##          "howmany: "+str(howmany)+
##          "center: "+str(center)+
##          "sigma: "+str(sigma)+
##          "sgl len: "+str(len(sgl))
##          );


##  good=[];

##ODO do it more nicely with weighted selection.

##  for i in range(howmany):
##      id=int(n.random.normal(center, sigma));
##      print(id);
##      good.append(sgl[id]);

#############
## COMPARE ##
#############

##ake them all same size so that histograms can be compared:

##  for i in range(len(good)):
##      if len(good[i].printPoints())<maxControlPoints:
##          for j in range (len(good[i].printPoints()), maxControlPoints):
##            #  print(good[i].printPoints());
##              #print(good[i].getFauxHistogram());
##              good[i].elevate();
##              print("Elevated "+str(i)+" to:", len(good[i].printPoints()));
##              print(good[i].printPoints());
##              #print(good[i].getFauxHistogram());
##            #  print(good[i].printPoints());
##      #print(good[i].getFauxHistogram());
##      good[i].graphMe("save", str(i)+":"+str(good[i].getRating())+":"+str(good[i].getLength()));

##  for i in range(len(good)):
##      print(str(len(good[i].printPoints()))+":"+str(i));


##  '''
##  print("Distance matrix:");

##  ds = n.empty(shape=(len(good), len(good)));
##  for i in range(len(good)):
##      for j in range(len(good)):
##          ds[i][j]=compare(good[i], good[j]);

##  print(ds);
##  '''



#############
## FIND SUCH SYMBOLS THAT THE SUM OF THEIR DISTANCES IS AS BIG AS POSSIBLE ##
#############

##https://stackoverflow.com/questions/43563475/find-n-symbols-as-unlike-to-each-other-as-possible-when-given-matrix-of-likene

##tart_time=time.time()
##  from pulp import *

##  def dist(i, j):
##      print("Comp."+str(i)+":"+str(j));
##      return compare(good[i], good[j]);

##  N = len(good)  # number of symbols
##  M = round(int(len(good)/2));  # number of symbols to select
##  print(str(M)+" are the symbols we are selecting, out of "+str(N));

##create problem object
##  prob = LpProblem("Greatest Distance Problem", LpMaximize)

##define decision variables
##  include = LpVariable.dicts("Include", range(N), 0, 1, LpInteger)
##  include_both = LpVariable.dicts("Include Both", combination(range(N), 2))

##add objective function
##  prob += (
##      lpSum([include_both[i, j]*dist(i, j) for i, j in combination(range(N), 2)]), 
##      "Total Distance"
##  )

##define constraints
##  prob += (
##      lpSum(include[i] for i in range(N)) == M,
##      "Select M symbols"
##  )
##  for i, j in combination(range(N), 2):
##      prob += (include_both[i, j] <= include[i], "")
##      prob += (include_both[i, j] <= include[j], "")

##iddle=time.time();
##  prob.solve()

##  print("Status: {}".format(LpStatus[prob.status]));
##  print("Included: {}".format([i for i in range(N) if value(include[i]) == 1]));

##nd=time.time();

##rint(">Middle: %s seconds ---" % (middle-start));
##rint(">Drawing %s seconds ---" % (end-start));

#############
## OUTPUT BEST SYMBOLS TO FOLDER ##
#############

##  print("Outputting the best ones to a diff folder");
##  print([i for i in range(N) if value(include[i]) == 1]);
##  for i in range(N):
##      if value(include[i]) == 1:
##          good[i].graphMe("save", "good/"+str(i)+":"+str(good[i].getRating())+":"+str(good[i].getLength()));

