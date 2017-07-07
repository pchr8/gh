import svgpathtools
import random;
from time import sleep
import bezier
import numpy as np
from math import hypot
import matplotlib.pyplot as plt
import seaborn
from math import atan2, degrees, pi

print("Testing");

def angle(x1, x2, y1, y2):
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return degs;
random.seed();

'''
# Coordinates are given as points in the complex plane
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier, Arc, disvg
seg1 = CubicBezier(300+100j, 100+100j, 200+200j, 200+300j)  # A cubic beginning at (300, 100) and ending at (200, 300)
seg2 = Line(200+300j, 250+350j)  # A line beginning at (200, 300) and ending at (250, 350)
path = Path(seg1, seg2)  # A path traversing the cubic and then the line

# We could alternatively created this Path object using a d-string
from svgpathtools import parse_path
path_alt = parse_path('M 300 100 C 100 100 200 200 200 300 L 250 350')

# Let's check that these two methods are equivalent
print(path)
print(path_alt)
print(path == path_alt)

# On a related note, the Path.d() method returns a Path object's d-string
print(path.d())
print(parse_path(path.d()) == path)

disvg(seg1);
sleep(1);
disvg(path);
sleep(1);
disvg(seg2);
seg1 = CubicBezier(300+200j, 100+100j, 200+200j, 200+300j)  # A cubic beginning at (300, 100) and ending at (200, 300)
sleep(1);
disvg(seg1);
seg1 = CubicBezier(300+200j, 100+200j, 200+200j, 200+300j)  # A cubic beginning at (300, 100) and ending at (200, 300)
sleep(1);
disvg(seg1);
'''

'''
nodes2 = np.array([
    [0.0 ,  0.0],
    [0.25,  2.0],
    [0.5 , -2.0],
    [0.75,  2.0],
    [1.0 ,  0.0],
])

nodes1 = np.array([
    [0.0 ,  0.0],
    [0.75,  2.0],
    [1.0 ,  0.0],
])


curve2 = bezier.Curve.from_nodes(nodes2)
curve1 = bezier.Curve.from_nodes(nodes1)

ax = curve2.plot(num_pts=256)
curve1.plot(num_pts=256, ax=ax)
plt.show();
'''

#Now let's generate something random.
for i in range(20):
    nOfPoints    = random.randint(2, 40);
    print(str(nOfPoints) +" control points")

#It can have three different places to start and three different places to end.
    start=random.randint(0, 2);
    end=random.randint(0, 2);
    print(start, end);

    rand = np.array(
            [[0, (10.0/6)*2*start+10.0/6]]
            )
    print(rand);

    for i in range (nOfPoints):  
        rpoints = [[random.randint(0, 10), random.randint(0, 10)]] ;
        print(rpoints);
        rand=np.append(rand, rpoints, axis=0);
    rand=np.append(rand,[[10,  (10.0/6)*2*end+10.0/6]], axis=0);

    #print("Final glyph:");
    #print(rand);

    randsym = bezier.Curve.from_nodes(rand)
    ax = randsym.plot(num_pts=256)
    plt.axis([0, 10, 0, 10]);


    #Now we need additional data, such as length down:

    needed=np.linspace(0.0, 1.0, nOfPoints*10);
    points=randsym.evaluate_multi(needed);
    distpoints=0;
    for i in range(1, len(points)):
        #print(points[i]);
        distpoints+=hypot(points[i][0]-points[i-1][0],points[i][1]-points[i-1][1]) 
        print(distpoints);
    print("Distance covered via points:"+str(distpoints));
    print("Length 'official':"+str(randsym.length));

    #Still with points, we can count the radical_changes_of_direction

    sumAngles=0;
    for i in range(1, len(points)):
        sumAngles+=angle(
                points[i][0], 
                points[i-1][0], 
                points[i][1], 
                points[i-1][1]
                );
        print(sumAngles);


    #plt.scatter(*zip(*points))
    #plt.plot(points);

    #Set a bit of nice graphics
    ym = plt.plot(0.1, 10.0/6, 'bs', 0.1, 10.0/6+(10.0/6)*2, 'bs',  0.1, 10.0/6+(10.0/6)*4, 'bs');
    yy=[10.0/6.0, 10.0/6.0];
    for a in range(0, 6, 2):
        lm = plt.plot([0, 10], [y+y*a for y in yy], linewidth=0.3, color='r');
        print([y*a for y in yy]);


    # And some settings!
    plt.text(2, 8, "length:"+str(randsym.length));
    plt.text(2, 7, "length via points;"+str(distpoints));
    plt.text(2, 6, "sum of angles"+str(sumAngles));

    #plt.show();
    plt.savefig(str(i)+"test.svg");
