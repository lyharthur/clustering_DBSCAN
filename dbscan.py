import numpy as np
import matplotlib.pyplot as plt
import math
import time

UNCLASSIFIED = False
NOISE = None

def dist_Manhatan(p,q):
    return np.absolute(p-q).sum() #Manhatan Distance

def dist_Euclidean(p,q):
    return math.sqrt(np.power(p-q,2).sum())#Euclidean Distance

def eps_neighborhood(p,q,eps):
    return dist_Manhatan(p,q) < eps

def region_query(array, point_id, eps):
    n_points = array.shape[1]
    seeds = []
    for i in range(0, n_points):
        if eps_neighborhood(array[:,point_id], array[:,i], eps):
            seeds.append(i)
    return seeds

def expand_cluster(array, classifications, point_id, cluster_id, eps, min_points):
    seeds = region_query(array, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = region_query(array, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
        
def dbscan(array, eps, min_points):
    
    #Inputs:
    #  array - shape(1,2)array
    #  eps - dist
    #  min_points - The minimum number of points to make a cluster
    #Outputs:
    #  cluster 1,2,3,4.... or None
    
    cluster_id = 1
    n_points = array.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        point = array[:,point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if expand_cluster(array, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications



def test_dbscan(filename):
    x = []
    y = []
    ans = []
    array = []
    with open(filename) as f:
        for line in f :
            line = line.replace('\n','')
            line = line.split('\t')      
            #print(float(line[0]),float(line[1]))
            x.append(float(line[0]))
            y.append(float(line[1]))
            ans.append(float(line[2]))
                
            #line = line.replace('\n','')
    array.append(x)
    array.append(y)
    array = np.array(array)

    ax = plt.gca()
    ax.set_facecolor('#F0F0F0')

    def drawplt_SSE(eps, min_points,cluster):
        o = open('output.txt','w')
        meanX = [0,0,0,0,0,0,0,0,0]
        meanY = [0,0,0,0,0,0,0,0,0]
        count = [0,0,0,0,0,0,0,0,0]
        SSE = [0,0,0,0,0,0,0,0,0]
        for index, pt in enumerate(cluster):
            if pt == 1:
                plt.plot(x[index],y[index],'b.')
                meanX[pt] += x[index]
                meanY[pt] += y[index]
                count[pt] += 1
            elif pt == 2:
                plt.plot(x[index],y[index],'r.')
                meanX[pt] += x[index]
                meanY[pt] += y[index]
                count[pt] += 1
            elif pt == 3:
                plt.plot(x[index],y[index],'g.')
                meanX[pt] += x[index]
                meanY[pt] += y[index]
                count[pt] += 1
            elif pt == 4:
                plt.plot(x[index],y[index],'c.')
                meanX[pt] += x[index]
                meanY[pt] += y[index]
                count[pt] += 1
            elif pt == 5:
                plt.plot(x[index],y[index],'m.')
                meanX[pt] += x[index]
                meanY[pt] += y[index]
                count[pt] += 1
            elif pt == 6:
                plt.plot(x[index],y[index],'y.')
                meanX[pt] += x[index]
                meanY[pt] += y[index]
                count[pt] += 1
            elif pt == 7:
                plt.plot(x[index],y[index],'k.')
                meanX[pt] += x[index]
                meanY[pt] += y[index]
                count[pt] += 1
            else:
                plt.plot(x[index],y[index],'w.')
                meanX[8] += x[index]
                meanY[8] += y[index]
                count[8] += 1

            o.write(str(x[index]) + ' ' + str(y[index]) + ' ' + str(pt)+'\n')
     
        filename = 'eps_'+ str(eps) +'&min_points_'+ str(min_points)+'.png'
        plt.title(filename)
        #plt.show()
        plt.savefig(filename,dpi=300,format="png")
        o.close()

        Dist = []
        for index, pt in enumerate(cluster):
            if pt !=None:
                distance2 = math.sqrt((meanX[pt]/count[pt] - x[index])**2 + (meanY[pt]/count[pt] - y[index])**2)**2
                SSE[pt] += distance2
            else:
                distance2 = math.sqrt((meanX[8]/count[8] - x[index])**2 + (meanY[8]/count[8] - y[index])**2)**2
                SSE[8] += distance2
                
        SSE_avg = 0
        c = 0
        for index,sse in enumerate(SSE):
            if sse!=0:
                print('cluster: '+ str(index) + ' -> SSE : ' + str(sse))
                SSE_avg +=  sse
                c += 1
        print('SSE_avg: ' + str(SSE_avg/c)) #sum of square error

    
##    def SSE():   
##        return numpy.sum(Dist**2)


    eps =50 
    min_points = 1
    
    cluster = dbscan(array, eps, min_points)
    drawplt_SSE(eps, min_points,cluster)
    
    
    
    #print(dbscan(m, eps, min_points))



##    for eps1000 in range(2200,3100):
##        for min_points in range(5,16):
##            eps = eps1000/1000
##            result = dbscan(array, eps, min_points)
##            #if (result.count(2) <= 280 and result.count(2) >= 272 ):
##            drawplt(eps, min_points,result)
##            print(eps, min_points)

    

if __name__ == '__main__':
    start = time.time()
    filename = 'clustering_test.txt'
    #clustering_test.txt (3.5 3)
    #spiral.txt  (2.5 3)
    #path_test.txt
    #flame.txt (1.64 10)
    #Aggregation.txt (2.2,2.3   11,12)
    #Compound.txt (1.5 9)?
    
    test_dbscan(filename)
    
    end = time.time()
    total = end - start
    print('--------------------done--------------------')
    print( "Total Time taken: ", total, "seconds.")
    
