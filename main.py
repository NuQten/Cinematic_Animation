


### CONSANTE ### 
Y_AC: float = 1.0 # Distance Y entre AC 
X_AC: float = 1.0 # Distance X entre AC 
X_CE: float = 1.0 # Distance X entre CE 

L_AB: float = 1.0 # Longeur semgent AB 
L_BD: float = 2.0 # Longeur semgent BD
L_DE: float = 3.0 # Longeur semgent DE




class Mechanism:

    def __init__(self, arg:list[int] = [2, 6, 4, 1, 3, 3]):  # noqa: B006
        self.r1 = arg[0]
        self.r2 = arg[1]
        self.r3 = arg[2]
        self.a = arg[3]
        self.b = arg[4]
        self.c = arg[5]

        self.fi0 = 0
        self.x0 = [3, 1, 5*pi/3, pi/3]

        self.sol = None
        self.res = {}
        # self.draw()


    def mech_sys(self, OUT, fi0):
    
        x, y, fi1, fi2 = OUT
    
        return [self.r1*cos(fi0) + x*cos(fi1) - self.b,
                self.r1*sin(fi0) + x*sin(fi1) + self.a,
                (self.r2 - x)*cos(fi1) + self.r3*cos(fi2) + self.c,
                (self.r2 - x)*sin(fi1) + self.r3*sin(fi2) - y]

    def sys_solve(self, fi0=None):
        if fi0 == None: 
            fi0 = self.fi0
        else:
            self.fi0 = fi0

        sol = optimize.root(self.mech_sys, self.x0, fi0)
        self.x0 = sol.x
        self.res = np.concatenate(([fi0], sol.x))

    def mech_PFS(self):
        pass


    def draw(self, ax = None) -> plt.Axes:
        if ax == None:
            _fig, ax = plt.subplots()
            flag = 1
        else : 
            flag = 0

        _x, y, fi1, _fi2 = self.x0
        A = [0, 0]
        B = [self.r1*cos(self.fi0), self.r1*sin(self.fi0)]
        D = [self.r1*cos(self.fi0) + self.r2*cos(fi1), self.r1*sin(self.fi0) + self.r2*sin(fi1)]
        E = [self.b + self.c, -self.a + y]

        ax.plot([A[0], B[0]], [A[1], B[1]])
        ax.plot([B[0], D[0]], [B[1], D[1]])
        ax.plot([D[0], E[0]], [D[1], E[1]])

        if flag == 1 : 
            ax.axis([-3, 8, -5, 8])
            plt.show()

        return ax

    def animate(self, fi0_min=0, fi0_max=2*pi, n_steps=360):
        fi0 = np.linspace(fi0_min, fi0_max, n_steps)

        _fig, ax = plt.subplots()

        ax.axis([-3, 8, -5, 8])
        

        for angle in fi0:
            self.sys_solve(angle)
            ax.cla()
            ax.set_xlim(-5, 10)
            ax.set_ylim(-7.5, 7.5)
            self.draw(ax)
            plt.pause(0.0001)
    

mech1 = Mechanism()
mech1.animate()




