from pylab import *
from numpy import *
from scipy.integrate import odeint
from scipy.optimize import minimize, fsolve
from scipy.sparse import diags


# Julius' example:

if False:
    import netCDF4 as nc
    
    filename = "garbe_example.nc"
    
    ncf = nc.Dataset(filename, "r") # read-only mode
    
    tsj = ncf.variables["time"][:] # extract time vector
    trajj = ncf.variables["slvol"][:].reshape((-1,1)) # extract sea-level volume vector
    
    ncf.close()
    


# parameter-induced tipping in standard hysteresis system:

dr = 1e-2  # very slow parameter change

sigma = 0 #.1 #.2  # noise level

plt = False

def dxr(xr, unused_t):
    x,r = xr
    return array([(-x**3 + x + r)/dr, 1])  # normal form of bifurcation

xr0 = [-1, 0]  # initially, x is a stable fixed point in the lower branch
ts1 = ts = linspace(0,1,100000)
if sigma == 0:
    traj = odeint(dxr, xr0, ts)
else:
    traj = zeros((ts.size,2))
    traj[0,:] = xr0
    for i in range(1,ts.size):
        dt = ts[i] - ts[i-1]
        traj[i,:] = traj[i-1,:] + dt * dxr(traj[i-1,:],NaN) 
        traj[i,0] += random.normal(scale=sqrt(dt)*sigma)
traj1 = traj
rcrit = sqrt(4/27)  # critical parameter value
tcrit1 = tcrit = ts[min(where(traj[:,1]>rcrit)[0])]  # time of tipping
print(tcrit)

if plt:
    figure()
    d1 = (traj[1:,0] - traj[:-1,0]) / (ts[1:] - ts[:-1])
    d2 = (d1[1:] - d1[:-1]) / (ts[2:] - ts[:-2])
    plot([tcrit,tcrit], [1e-5,10], "k:")  # tipping marker
    semilogy(ts[1:], d1, label="dx/dt")  # numerical derivative
    #plot(ts[2:], d2, label="d¹x/dt²")  # numerical derivative
    title("x.=-x³+x+r, r.="+str(dr))
    show()

# notes:
# initially superexponential growth
# after turning point very fast shift to new level
# followed by much slower growth than before


# parameter-induced tipping in alternative system:

def dxr2(xr, unused_t):
    x,r = xr
    return array([(-x * (r + (x+1)**2))/dr, 1])

xr0 = [-2, -1]  # initially, x is a stable fixed point in the lower branch
ts2 = ts = linspace(0,2,100000)
if sigma == 0:
    traj = odeint(dxr2, xr0, ts)
else:
    traj = zeros((ts.size,2))
    traj[0,:] = xr0
    for i in range(1,ts.size):
        dt = ts[i] - ts[i-1]
        traj[i,:] = traj[i-1,:] + dt * dxr2(traj[i-1,:],NaN) 
        traj[i,0] += random.normal(scale=sqrt(dt)*sigma)
traj2 = traj 
rcrit = 0  # critical parameter value
tcrit2 = tcrit = ts[min(where(traj[:,1]>rcrit)[0])]  # time of tipping
print(tcrit)

if plt:
    figure()
    d0 = traj[:,0]
    d1 = (d0[1:] - d0[:-1]) / (ts[1:] - ts[:-1])
    d2 = (d1[1:] - d1[:-1]) / (ts[2:] - ts[:-2])
    subplot(411)
    plot([tcrit,tcrit], [-2,2], "k:")  # tipping marker
    plot(ts, d0, label="x")
    subplot(412)
    plot([tcrit,tcrit], [1e-5,.1], "k:")  # tipping marker
    plot(ts[1:], d1, label="dx/dt")  # numerical derivative
    subplot(413)
    plot([tcrit,tcrit], [1e-5,.1], "k:")  # tipping marker
    semilogy(ts[1:], d1, label="dx/dt")  # numerical derivative
    subplot(414)
    plot([tcrit,tcrit], [-.01,.01], "k:")  # tipping marker
    plot(ts[2:], d2, label="d¹x/dt²")  # numerical derivative
    title("x.=-x(r+(x+1)²), r.="+str(dr))
    show()


# parameter-induced tipping in macro-descr. of network system:

def dxr3(xr, unused_t):
    x,r = xr
    p = 0.03
    return array([(p * (r * (1 - x) - (1 - r) * x) + (1 - p) * x * (1 - x) * (x * r - (1 - x) * (1 - r)))/dr, 1])

xr0 = [1e-10, 0]  # initially, x is a close to a stable fixed point in the lower branch
ts3 = ts = linspace(0,1,100000)
if sigma == 0:
    traj = odeint(dxr3, xr0, ts)
else:
    traj = zeros((ts.size,2))
    traj[0,:] = xr0
    for i in range(1,ts.size):
        dt = ts[i] - ts[i-1]
        traj[i,:] = traj[i-1,:] + dt * dxr3(traj[i-1,:],NaN) 
        traj[i,0] += random.normal(scale=sqrt(dt)*sigma)
traj3 = traj
rcrit = .287  # critical parameter value ??????
tcrit3 = tcrit = ts[min(where(traj[:,1]>rcrit)[0])]  # time of tipping
print(tcrit)

if plt:
    figure()
    d1 = (traj[1:,0] - traj[:-1,0]) / (ts[1:] - ts[:-1])
    d2 = (d1[1:] - d1[:-1]) / (ts[2:] - ts[:-2])
    plot([tcrit,tcrit], [1e-5,10], "k:")  # tipping marker
    semilogy(ts[1:], d1, label="dx/dt")  # numerical derivative
    title("x.=p(r(1-x)-(1-r)x)+(1-p)x(1-x)(xr-(1-x)(1-r)), r.="+str(dr))
    show()


# logistic curve as "internal" sigmoid-shaped dynamics for comparison:

def dx(x, unused_t):
    return abs(x * (1-x))

x0 = 1e-5
ts4 = ts = linspace(0,20,100000)
if sigma == 0:
    traj = odeint(dx, x0, ts)
else:
    traj = zeros((ts.size,1))
    traj[0] = x0
    for i in range(1,ts.size):
        dt = ts[i] - ts[i-1]
        traj[i] = traj[i-1] + dt * dx(traj[i-1],NaN) 
        traj[i] += random.normal(scale=sqrt(dt)*sigma)
traj4 = traj
tcrit4 = NaN
if plt:
    figure()
    d1 = (traj[1:,0] - traj[:-1,0]) / (ts[1:] - ts[:-1])
    d2 = (d1[1:] - d1[:-1]) / (ts[2:] - ts[:-2])
    semilogy(ts[1:], d1, label="dx/dt")  # numerical derivative
    plot(ts[2:], d2, label="d¹x/dt²")  # numerical derivative
    title("x.=x(1-x)")
    show()


def fit(x, pos, bw, *, dx=None, future=True):
    """
    fit a polynomial to dx(x) at position pos,
    using a Gaussian kernel with bandwidth bw  
    """
    if dx is None:
        dx = x[1:] - x[:-1]
        x = (x[1:] + x[:-1]) / 2
        pos = pos - 0.5
    n = len(x)
    index = arange(n)
    relindex = index - pos
    x0 = x[int(pos)] if pos%1 == 0 else pos%1 * x[int(ceil(pos))] + (1-pos%1) * x[int(floor(pos))]
    relx1 = x - x0
    relx2 = relx1**2
    relx3 = relx1**3

    # dieser block dient dazu die Gewichtung (kernel) zu bauen.
    ddx = dx[1:] - dx[:-1]
    dddx = 0*dx
    dddx[1:-1] = ddx[1:] - ddx[:-1]
    weight0 = 1 / (1 + 9*(2*relindex/bw)**4)
    if not future: weight0[int(ceil(pos)):] = 0
    vardx = average(dx**2, weights=weight0) - average(dx, weights=weight0)**2
    vardddx = average(dddx**2, weights=weight0) - average(dddx, weights=weight0)**2
    print(vardx,vardddx)
    weight = 1 / (1 + 9*(relindex**4 + vardx/vardddx)*(2/bw)**4)
    if not future: weight[int(ceil(pos)):] = 0
    weight /= sum(weight)
    ####################################################
    def negloglikelihood(theta):
        A,a,B,b,C,c,D,d,S,s = theta
        predicted_dx = (A + a*relindex) + (B + b*relindex)*relx1 + (C + c*relindex)*relx2 + (D + d*relindex)*relx3
        logvar = S + s*relindex
        return average((predicted_dx - dx)**2 / 2 / exp(logvar) + logvar / 2, weights=weight)
    # use motabar to get first estimate:
    W = diags(weight)
    X = array([0*relx1+1,relindex,relx1,relindex*relx1,relx2,relindex*relx2,relx3,relindex*relx3]).T
    Pphi = W.dot(X).T.dot(X)
    muphi = inv(Pphi).dot((W.dot(dx).T.dot(X)).T)
    theta = concatenate((muphi,[log(vardx),0]))
    nll = negloglikelihood(theta)
    print(theta, nll)
    # minimize loglikelihood with changing noise level:
    res = minimize(negloglikelihood, theta, 
                   method='Powell'  # CG/BFGS/TNC slower, COBYLA very slow, Nelder-Mead/L-BFGS-B/SLSQP fail
                   )
    if not res['success']: 
        print("uups",pos)
    theta = res['x']
    nll = res['fun']
    print(theta, nll)
    A,a,B,b,C,c,D,d,S,s = theta
    # find closest bifurcation point:
    def zero(y):
        relindex, relx1 = y
        relx2 = relx1**2
        relx3 = relx1**3
        F = (A + a*relindex) + (B + b*relindex)*relx1 + (C + c*relindex)*relx2 + (D + d*relindex)*relx3
        dF = (B + b*relindex) + 2*(C + c*relindex)*relx1 + 3*(D + d*relindex)*relx2
        return [F, dF]
    critrelindices, critrelxs = array([fsolve(zero, [-n, 0]), fsolve(zero, [0, 0]), fsolve(zero, [n, 0])]).T
    wh = argmin(abs(critrelindices))
    critrelindex, critrelx = critrelindices[wh], critrelxs[wh]
    critpos = critrelindex + pos
    critx = critrelx + x0
    return {'pos':pos, 'relx':relx1, 'theta':theta, 'likelihood':exp(-nll), 'critpos':critpos, 'critx':critx, 'weights':weight}




#traj, tcrit, ts, inds = trajj, nan, tsj, arange(5*tsj.size//8,23*tsj.size//32,tsj.size//1000,dtype="int")
#traj, tcrit, ts, inds = traj1, tcrit1, ts1, arange(10000,60000,1000)
#traj, tcrit, ts, inds = traj2, tcrit2, ts2, arange(10000,60000,1000)
traj, tcrit, ts, inds = traj3, tcrit3, ts3, arange(10000,60000,1000)  # works less well
#traj, tcrit, ts, inds = traj4, tcrit4, ts4, arange(10000,60000,1000)
res = [fit(traj[:,0],pos,20000,future=True) for pos in inds]
like = array([res[i]['likelihood'] for i in range(len(res))])
f, (ax1,ax2) = subplots(2, sharex=True)
ax1.plot(ts, 
         traj[:,0])
ax1.plot(ts[1:], 
         res[-1]['weights']/max(res[-1]['weights']))
colors = zeros((len(inds),4))
colors[:,3] = like / max(like)
predictions = array([res[i]['critpos'] for i in range(len(res))]) * (max(ts) - min(ts)) / len(ts) + min(ts)
print(predictions)
ax2.scatter([ts[i] for i in inds], 
            predictions, 
            color=colors)
ax2.plot(ts, 
         0*ts + tcrit, "g:")
ax2.plot(zeros(2)+tcrit, [ts[0],ts[-1]], "g:")
ax2.plot([ts[0],ts[-1]], [ts[0],ts[-1]], "g")
ax2.set_ylim(ts[0],ts[-1])
show()
