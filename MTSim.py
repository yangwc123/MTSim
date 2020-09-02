### MTS --- Multiple trapping simulation ###
###             (c) J. Lorrmann           ###
### Commented by J. Gorenflot in November 2012
### Modified by J. Gorenflot in March 2013 to add a T dependence of k_br
### Modified by W. Yang in January 2019 
### in the test function
#

#from progressbar import SingleBar, ProgressBar, FormatLabel, Counter, ETA      #Looks like they're not used
import sys
# sys.path.append(r"C:\Users\YANGW0C\desktop\MTRsimulation\LPlot")
#sys.path.append(".\LPlot")
import numpy as np
from math import *

from scipy import integrate, Inf
import weave
from scipy.optimize import fsolve, fmin
from LPlot import LP_data
#from material import Material

#from progressbar import SingleBar, ProgressBar, FormatLabel, Counter, ETA      #Looks like they're not used

class const:
    """This class provides physical constants
    """
    from scipy import constants as cons
    q = cons.physical_constants['elementary charge'][0]
    k_eV = cons.physical_constants['Boltzmann constant in eV/K'][0]
    eps0 = cons.physical_constants['electric constant'][0]
    pi = cons.pi


### simulation ###

class Simulation:
    """ Simulates the bimolecular recombination of an inial density of charges.

    Attributes:
    tmin:   Starting simulation time (measured from the instant of generation
    of the initial density of charges)
            default : 1e-11 s
    tmax:   Ending simulation time.
            default : 1e-2 s
    nt:     Number of points in the time axis,
    logarithmically distributed between tmin and tmax.
            default : 100
    Emin:   Deepest considered energy level.
            default : -1 eV
            NoteJG  : Why integer? 
    Emax:   Highest considered energy level.
            default : 1 eV; forced to 0.0 eV in case of exponential DOS
            NoteJG  : Why integer? 
    nE:     Number of energetical levels of the discretized DOS(?)
            default : 40
    sigma:  Standard deviation in case of Gaussian DOS, ref Energy for an exponential DOS
            default : 0.1 eV
    T:      Temperature.
            default : 300.0K
    F:      External electric field(?)
            default : 1.0e7 V/m
    A:      ??
            default : 0.0
    gamma:  Tuneling facor(?), used for the calculation of tau from nu
            default : 5e9 m-1
    a:      Intersite distance, used for calculating tau from nu
            default : 1e-9 m
    nu0:    Attempt-to-escape frequency
            default : 1e12 s-1
    Nt:     Total density of trap states (?)
            default : 1e27 m-3
    nn:     Initial density of charges (?)
            default : 1e22 m-3
    kbr:    ??
            default : 1e-25
    zeta:   ??
            default : 0.0
    d:      distance between free sites (?)
            default : 1e-7 m
            # NoteJG: What's the diff with a? NoteJG: not used
    G:      Generation rate?
            default : 0.0
    Eg:     ?? NoteJG: not used
            default : 1.7
    DOS:    Density of states profile: 'exp' (or 'EXP', 'eXP', 'Exp',...) -> exponential. Otherwise: Gaussian
            default : 'exp' 
    pprint: ??
            default : 'False'
    gen:    Initial distribution: 'con' -> all at zero energy (con for conductive), else = following the DOS
            default :'con'

    Functions:
    __init__(tmin, tmax, nt, Emin, Emax, nE, sigma,  T, F, A, gamma, a, nu0, Nt, nn, kbr, zeta, d,  G, Eg, DOS, pprint, gen):
                     clear.
    calc(t):         Solves the rate equations using integrate.odeint
                     NoteJG: probably outdated: it does not call the write functions to define the rate equations
    solve(te):       Solves the rate equations using integrate.ode
    __rate_eq(N, t, res, g, size, tr, rel, tau, Nt, G, kbr, zeta):
                     Creates the string of the C++ code to calculates recombination, trapping and release rates
                     and returns dn(E)/dt rates for a given distribution of charges (see the doc of the function)
    __rate_eq_solve(N, t, res, g, size, tr, rel, tau, Nt, G, kbr, zeta):
                     executes the C++ string defined by __rate_eq(?)
    calc_rates():    Calculates the trapping and recombination
                     NoteJG: possible error for the trapping rate
    dataExtraction():?    
    

    Additionally initialized in __init__:
    t:         time axis (logarithmic distribution of nt points between tmin and tmax)
    dE:        discretized energy step (possible mistake in case of gaussian distribution)
    tau0:      intrinsic trapping time constant (claculated from attempt to escape frequency)
    tau:       d, T and F dependent trapping time constant
    nu:        matrix of transfer rates between energy levels (?)
    E:         array of discretized energy levels (NoteJG: the first definition seems to me 1/strange, 2/useless,maybe it's historical remain)
    g:         density of states (exponential or gaussian), array of size 2nE... (Note JG: why 2 times nE? Note WY: maybe for Gaussian the tail states are doubled)
    calc_n:    function calculatinf the density of occupied states (using g and Fermi distribution)
    n:         matrix of density of charges (energy level, time)
    rates:     recomb rates??? initialized to zero
    initial:   initial energy distribution of the charges, following g or having all (lowest?) level depending on the value of 'gen'
    tr:        trapping rate = 1/Nt/tau0 (NoteJG: shouldn't it be Nt/tau0?)
    rel:       release rate (tr and rel are initialized by calling calc_rates())
   
    
    """

    def __init__( self, tmin = 1e-11, tmax = 1e-2, nt = 100, Emin = -1, Emax = 1,
                 nE = 40, sigma = 0.1,  T = 300.0, F = 1.0e7, A = 0.0, gamma = 5e9,
                 a = 1e-9, nu0 = 1e12, Nt = 1e27, nn = 1e22, kbr = 1e-25, zeta = 0.0,
                 d = 1e-7,  G = 0.0, Eg = 1.7, DOS = 'exp', pprint = False, gen = 'con' ):

        if DOS.lower() == 'exp':
            Emax = 0.0

        self.t = np.logspace( np.log10( tmin ), np.log10( tmax ), nt )
        self.pprint = pprint
        self.gamma = gamma
        self.a = a
        self.T = T
        self.F = F
        self.A = A
        self.nu0 = nu0
        self.Nt = Nt
        self.nn = nn
        self.sigma = sigma
        self.kbr = kbr
        self.zeta = zeta
        self.G = float(G)
        self.Eg = Eg
        self.dE = float(( Emax - Emin ) / nE)
        # Energetical step between successive levels
        # NoteJG : I fear that's an integer.
        self.nE = nE
        self.gen = gen

        self.tau0 = 1. / ( self.nu0 * exp( -2 * self.a * self.gamma) )

        self.tau = d / F * const.k_eV * T / a**2 * self.tau0

        self.nu = np.zeros((nE, nE), dtype=np.float)
        self.E = np.zeros((2*nE,), dtype=np.float)
        #NoteJG: why 2 times (nE)?
        self.E = np.linspace( Emin, Emax, nE )


        if DOS.lower() == 'exp':
            # exponential DOS (g -> DOS; calc_n -> DOOS, with Fermi-Dirac distribution)
            self.g = np.zeros((2*nE,), dtype=np.float)
            self.g = Nt / sigma * np.exp( self.E  / sigma ) * self.dE
            self.calc_n = lambda E, Ef, theta: theta*self.Nt/self.sigma * exp( E / self.sigma )/ ( 1 + exp( (E-Ef ) / ( const.k_eV * self.T ) ) )
        else:
            # gaussian DOS (g -> DOS; calc_n -> DOOS, with Fermi-Dirac distribution)
            self.g = Nt / ( sqrt( 2 * const.pi ) * sigma ) * np.exp(- (self.E)**2 / (2 * sigma**2) ) *  self.dE
            self.calc_n = lambda E, Ef, theta: theta*self.Nt/(sqrt(2*const.pi)*self.sigma ) * exp(- (E)**2 / (2*self.sigma**2))/ ( 1 + exp( (E-Ef ) / ( const.k_eV * self.T ) ) )


        # Results
        self.n = np.zeros((nE, nt), dtype=np.float)
        self.rates = np.zeros((nE, nt), dtype=np.float)

        if gen.lower() == 'con':
            self.initial = np.zeros(self.E.shape[0])
            self.initial[-1:] = nn
            # NoteJG: why the last one? because conductive charges are considered to be the ones with "zero" energy,
            # Which is assumed to be the last energy of the serie. Doesn't work for a Gaussian going between -1eV and +1eV
        else:
            self.initial = self.g * nn / Nt

        self.calc_rates()
        # NoteJG: defines "tr" and "rel": trapping and relaxation rates, respectively

    def calc( self, t = None ):
        ## Solve the rate equation
        if t is not None:
            self.t = t

        params = (np.zeros(self.E.shape[0]), self.g, self.E.shape[0], self.tr,
                  self.rel, self.tau, self.Nt, self.G, self.kbr, self.zeta)

        res, suc = integrate.odeint(self.__rate_eq, self.initial, self.t, args = params,
                                 printmessg = False, full_output = 1, mxhnil = 1, mxstep = 10000)

        self.n = res.transpose()
        self.ne = None
        # NoteJG: Who's that? and what's the use of it
        if 'successful' in suc['message']:
            self.dataExtraction()

    def solve( self, te=100 ):
        ## Solve the rate equation
        params = (np.zeros(self.E.shape[0]),self.g, self.E.shape[0], self.tr,
                  self.rel, self.tau, self.Nt, self.G, self.kbr, self.zeta)

        r = integrate.ode(self.__rate_eq_solve).set_integrator('vode', method='bdf',nsteps=30000)

        r.set_initial_value(self.initial, 0.0).set_f_params(*params)
        n = 1e-300
        i = 0
        j = 0
        while r.successful() and r.t < self.t[-1]:
            r.integrate(self.t[j], relax=True)
#            if np.all((1-r.y/n)**2) < 1e-8:
#                print 'BREAKING...', r.t
#                break
            n = r.y
            self.n[:,j] = n
            j+=1

        self.ne = n
        self.dataExtraction()

    def __rate_eq( self, N, t, res, g, size, tr, rel, tau, Nt, G, kbr, zeta):
        ''' Rates equations in C++
        Imputs:
        N:    density of charges (array, one value for one energy -> N(size-1) = density of free charges)
        t:    time
              #noteJG: doesn't seem to play a role
        res:  dn(E)/dt except that the last value is thesum of dn/dt
        g:    DOS 
        size: number of considered energy levels
        tr:   trapping rate (independent of E)
        rel:  relaxation rate (E)
        tau:  monomol rec. time?
        Nt:   total density of traps
        G:    generation rate
        kbr:  kbr
        zeta: proportion of recombining trapped charges. zeta = 0 -> trapped charges don't recombine.
        '''
        code = r'''
                using namespace std;
                int k;
                double tmp_res = 0.0;
                double _rec,  _rel, _tr;

                // OMP parallel for-loops crashes on my computer (???)
                //#pragma omp parallel for private(k,_rec,_tr,_rel) shared(res, N, g)
                //NoteJG: loop = trapped charges, out of loop = free charges
                //NoteJG: it means that free charges are defined as the ones with the last energy.
                //NoteJG: Which implies that the last energy HAS to be zero, even in the Gaussian DOS
                //NoteJG: (and not go until +1eV as proposed in the default parameters)
                for (k = 0; k < size - 1; k++ )
                {
                    _rec = zeta * kbr * max(N(k), 0.0) * max(N(size-1), 0.0);
                    _tr = tr * max( g(k) - max(N(k), 0.0), 0.0 ) * max(N(size-1), 0.0);
                    _rel = rel(k) * max(N(k), 0.0);

                    res(k) = _tr - _rel - _rec;// + G * g(k)/Nt;
                    tmp_res = tmp_res + (-_tr + _rel - _rec);
                }

                res(size-1) = tmp_res - kbr * max(N(size-1), 0.0) * max(N(size-1), 0.0) - 1/tau * max(N(size-1), 0.0) + G;
        '''
        import os
        #noteJG, not sure it's still quite usefull to import it
        #os.environ['CC'] = '/usr/local/bin/gcc'
        #os.environ['CXX'] = '/usr/local/bin/g++'
        err = weave.inline(code,
                            ['N', 'g', 'size', 'res', 'tr', 'rel',
                             'tau', 'Nt', 'G','kbr','zeta'],
                            type_converters=weave.converters.blitz,
                            compiler='gcc',
                            extra_compile_args =['-O3 -fopenmp -Wall -c -fmessage-length=0',
                                                 '-fno-strict-aliasing -w -pipe -fwrapv -Wall -fPIC'],
                            extra_link_args=['-lgomp -lm -lgomp -lpthread -fPIC'],  #we removed -ldl and -lutil
                            headers = ['<stdio.h>', '<algorithm>','<omp.h>'])#,
                            #include_dirs=["/usr/linclude/c++/4.2.1/"])
        return res


    def __rate_eq_solve( self, t, N, res, g, size, tr, rel, tau, Nt, G, kbr, zeta):
         return self.__rate_eq(N, t, res, g, size, tr, rel, tau, Nt, G, kbr, zeta)


    def calc_rates( self ):
        self.tr = 1 / self.tau0 / self.Nt
        self.rel = 1 / self.tau0 * np.exp(( self.E  - self.a * self.F) / ( const.k_eV * self.T ) )


    def dataExtraction( self ):
        from LPlot import LP_func

        if self.ne is not None:
            self.n = np.array([np.where(self.n[i]==0, self.ne[i], self.n[i]) for i in xrange(self.E.shape[0])])
        else:
            self.ne = self.n[:,-1]

        self.dndt = np.array([LP_func.derive_lin(self.t, self.n[i, :]) for i in xrange(self.E.shape[0])])

        self.nu_total = np.array([ ( np.trapz( self.rel[:-1] * self.n[:-1,i], self.E[:-1]) + 1/self.tau0 * self.n[-1,i] )/ np.trapz( self.n[:,i], self.E)
                                             for i in xrange(self.t.shape[0])])

        self.n_total = np.sum(self.n, axis=0)
        self.dn_totaldt = LP_func.derive_lin( self.t, self.n_total )

        self.ninv = np.array([self.g - self.n[:,i] for i in xrange(self.t.shape[0])]).transpose()

        self.n0 = self.n_total[0]


        self.nc0 = self.n[-1,0]

        self.nc = self.n[-1]

        self.nce = self.ne[-1]

        self.Ef = self.E[self.n[:-1,-1].argmax()]

        self.theta = np.array([self.n[:,i]/self.g for i in xrange(self.t.shape[0])]).transpose()

    def todataObject(self):
        do = LP_data.m_dataObject()
        for k,v in self.__dict__.items():
            if not hasattr(v, '__call__'):
                if np.asarray(v).shape != ():
                    do.setData(k, v)
                else:
                    do.setHeader(k, v)

        return do


## MAIN ###
    # noteJG: original settings were:
    #(nn=1e22, F=1e-300, Nt=1e27, nE=100, Emin=-1.0, sigma=0.069, Emax=0.0,
    #                     T=float(T), DOS=DOS, nt=100, tmax=1, tmin=1e-9,
    #                     kbr = kbr, zeta = zeta, gen=gen, G=0.0)
       
def test(Tf, kbr0 = 4.0e-15, zeta = 0.5, DOS = 'exp', gen = 'all' ):
    # import phc
    dh = LP_data.m_dataHandler()
    dOs=[]
    if DOS.lower() == 'exp':
        sigma = 0.045
    else:
        sigma=0.0366

    for T in Tf:
        if DOS.lower() == 'exp':
            kbr = kbr0 * exp(-sigma/T/const.k_eV)        # prefactor extracted from the exponential fit
                                                                # of the transients (see file "ExpKbr")
#            kbr = kbr / exp(-sigma/300/const.k_eV)*exp(-0.075/300/const.k_eV)   #just so that the value at 300K remains comparable
        else:
            kbr = 4.2e-16 * exp(-(2*sigma/T/3/const.k_eV)**2)# prefactor extracted from the Gaussian fit of
                                                            # the transients (see file "ExpKbr")

        sim = Simulation(nn=1e24, F=1e-300, Nt=1e27, nE=100, Emin=-1.0, sigma = sigma, Emax=0.0,
                         T=float(T), DOS=DOS, nt=100, tmax=1, tmin=1e-9,
                         kbr = kbr, zeta = zeta, gen=gen, G=0.0)
        sim.calc()
        dOs.append(sim.todataObject())
        print('Finished: T = ', T)
        sys.stdout.flush()

    dh.add(dOs)

    return dh


def process( d0 ):
        from LPlot import LP_func

        E = d0.getData('E')
        t = d0.getData('t')
        n = d0.getata('n')
        E_mean = np.array([ ( np.trapz(E * n[:,i], E))/ np.trapz( n[:,i], E)
                                             for i in xrange(t.shape[0])])

        d0.setData('Emean', E_mean)

