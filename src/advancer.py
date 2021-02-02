"""

.. module:: advancer
   :platform: MacOS, Unix
   :synopsis: A useful module indeed.

.. moduleauthor:: SHB <spencer@caltech.edu> and Esteban Cisneros <csnrsgr2@illinois.edu>


"""

import sys

sys.path.append("../utils/")

from stats_util import *
from pretty_print_util import *
from simulation_domain import *
import numpy as np
import csv
import h5py


class time_advancer:
    """This class advances the moment transport equations in time.

    A ``config`` dictionary is required to usethis class.
    Example files are provided in ``inputs/``.
    The dictionary is used to set the following variables:

    :ivar method: Time integration scheme (``euler``, ``RK2, or ``RK3``)
    :ivar time_step: Time integration time step
    :ivar final_time: Final integration time
    :ivar num_steps: Number of integration steps
    :ivar num_steps_print: Report-status frequency
    :ivar num_steps_write: Write-to-file frequency
    :ivar output_dir: Directory to which output is written
    :ivar output_id: An ID for each run's output file

    Create an advancer object simply by typing

    >>> advancer = time_advancer( config )

    Before running, you'll need to initialize the state. One way you may do this is through:

    >>> advancer.initialize_state_gaussian_univar( mu, sigma )

    for a 1D problem, where ``mu`` and ``sigma`` are user-specified. Now all you have to do is:

    >>> advancer.run()
    """

    def __init__(self, config):
        """Constructor

        :param config: Configuration
        :type config: dict

        """
                
        self.method          = config['advancer']['method']
        self.time_step       = config['advancer']['time_step']
        self.final_time      = config['advancer']['final_time']
        self.num_steps       = config['advancer']['num_steps']
        self.num_steps_print = config['advancer']['num_steps_print']
        self.num_steps_write = config['advancer']['num_steps_write']
        self.output_dir      = config['advancer']['output_dir']
        self.output_id       = config['advancer']['output_id']
        self.write_to        = config['advancer']['write_to']
        
        self.domain = simulation_domain(config)

        if self.method == "Euler":
            self.advance = self.advance_euler
            self.num_stages = 1
        elif self.method == "RK2":
            self.advance = self.advance_RK2
            self.num_stages = 2
        elif self.method == "RK23":
            self.advance = self.advance_RK23
            self.num_stages = 3

        num_points = self.domain.num_points
        num_moments = self.domain.qbmm_mgr.num_moments
        self.state = np.zeros([num_points, num_moments])
        self.rhs = np.zeros([num_points, num_moments])
        self.stage_state = np.zeros([self.num_stages, num_points, num_moments])
        self.stage_k = np.zeros([self.num_stages, num_points, num_moments])
        
        self.max_time_step = 1.e5
        self.min_time_step = self.time_step

        if "error_tol" in config["advancer"]:
            if self.method == "Euler":
                print("Euler method is non-adaptive. Ignoring error_tol.")
                self.adaptive = False
            elif self.method == "RK2":
                print("To run RK2 adaptively, specify cfl, ignoring error_tol.")
                self.adaptive = False
            else:
                self.adaptive = True
                self.error_tol = config["advancer"]["error_tol"]
                self.adapt_time_step = self.adapt_time_step_RK23
        else:
            self.adaptive = False

        if "cfl" in config["advancer"]:
            if self.method == "RK23":
                print("RK23 adapts time step per error_tol, ignoring cfl")
            else:
                self.cfl = config["advancer"]["cfl"]
                self.adaptive = True
                self.adapt_time_step = self.adapt_time_step_cfl
        else:
            self.cfl = 1
            
        # Report
        print('advancer: init: Configuration options ready')
        print('\t method          = %s'   % self.method)
        print('\t time_step       = %.4E' % self.time_step)
        print('\t final_time      = %.4E' % self.final_time)
        #print('\t error_tol       = %.4E' % self.error_tol)
        print('\t num_steps_print = %i'   % self.num_steps_print)
        print('\t num_steps_write = %i'   % self.num_steps_write)
        print('\t output_dir      = %s'   % self.output_dir)
        print('\t output_id       = %s'   % self.output_id)
        print('\t write_to        = %s'   % self.write_to)

        # Output options
        self.file_name = self.output_dir + 'qbmm_state_' + self.output_id
        if self.write_to == 'txt':
            self.file_name += '.dat'
            self.write_to_file = self.write_to_txt
        elif self.write_to == "h5":
            self.file_name += ".h5"
            self.write_to_file = self.write_to_h5
        return

    def initialize_state(self, init_state):
        """
        This function initializes the advancer state

        :param init_state: Initial condition
        :type init_state: array like
        """

        self.state = init_state

        print("state", self.state)

        return

    def initialize_state_jets(self):
        """
        This function intializes the advancer state to R.O. Fox's jet conditions
        """
        self.domain.initialize_state_jets(self.state)
        return
    
    def initialize_state_gaussian_trivar(self, mu1, mu2, mu3, sig1, sig2, sig3):
        """
        This function initializes the state to the raw moments of a trivariate Gaussian distribution.

        :param mu1: Mean of coordinate 1
        :param mu2: Mean of coordinate 2
        :param mu3: Mean of coordinate 3
        :param sig1: Standard deviation of coordinate 1
        :param sig2: Standard deviation of coordinate 2
        :param sig3: Standard deviation of coordinate 3
        :type mu1: float
        :type mu2: float
        :type mu3: float
        :type sig1: float
        :type sig2: float
        :type sig3: float
        """

        self.state[:] = raw_gaussian_moments_trivar(self.qbmm_mgr.indices, mu1, mu2, mu3, sig1, sig2, sig3)
        message = 'advancer: initialize_trigaussian: '
        f_array_pretty_print(message, 'state', self.state)
        return

    def initialize_state_gaussian_bivar(self, mu1, mu2, sigma1, sigma2):
        """
        This function initializes the state to the raw moments of a bivariate Gaussian distribution.

        :param mu1: Mean of coordinate 1
        :param mu2: Mean of coordinate 2
        :param sig1: Standard deviation of coordinate 1
        :param sig2: Standard deviation of coordinate 2
        :type mu1: float
        :type mu2: float
        :type sig1: float
        :type sig2: float
        """
        self.state[:] = raw_gaussian_moments_bivar(self.qbmm_mgr.indices, mu1, mu2, sigma1, sigma2)
        message = 'advancer: initialize_bigaussian: '
        f_array_pretty_print(message, 'state', self.state)
        return

    def initialize_state_gaussian_univar(self, mu, sigma):
        """
        This function initializes the state to the raw moments of a univariate Gaussian distribution.

        :param mu: Mean
        :param sigma: Standard deviation
        """

        self.state[:] = raw_gaussian_moments_univar(self.num_dim, mu, sigma)
        message = "advancer: initialize_gaussian: "
        f_array_pretty_print(message, "state", self.state)
        return

    def initialize_flow(self):
        """
        This function initializes flow
        """
        self.simulation_domain
        return
    
    def advance_euler(self):
        """
        This function advances the state with an explicit Euler scheme
        """
        # Stage 1: { y_1, k_1 } = f( t_n, y_0 )
        self.stage_state[0] = self.state.copy()
        self.stage_k[0] = self.domain.compute_rhs(self.stage_state[0])
            
        # Updates
        self.state = self.stage_state[0] + self.time_step * self.stage_k[0]
        self.domain.update_quadrature(self.state)
        self.time_step = self.cfl * self.domain.grid_spacing / self.domain.max_abscissa()

        return


    def advance_RK2(self):
        """
        This function advances the state with a Runge--Kutta 2 scheme
        """

        # Stage 1:
        self.stage_state[0] = self.state.copy()
        self.stage_k[0] = self.domain.compute_rhs(self.stage_state[0])
        self.stage_state[1] = self.stage_state[0] + self.time_step * self.stage_k[0]

        # Update domain quadratures
        self.domain.update_quadrature(self.stage_state[1])
        
        # Stage 2:
        self.stage_k[1] = self.domain.compute_rhs(self.stage_state[1])
        
        # Update
        self.state = 0.5*(self.stage_state[0] + self.stage_state[1]
                          + self.time_step*self.stage_k[1])

        # Update domain quadratures
        self.domain.update_quadrature(self.state)

        return
    
    def advance_RK3(self):
        """
        This function advances the state with a Runge--Kutta 2/3 scheme
        """

        # Stage 1: { y_1, k_1 } = f( t_n, y_0 )
        self.stage_state[0] = self.state.copy()
        self.qbmm_mgr.compute_rhs(self.stage_state[0], self.stage_k[0])
        self.stage_state[1] = self.stage_state[0] + self.time_step * self.stage_k[0]

        # Stage 2: { y_2, k_2 } = f( t_n, y_1 + dt * k_1 )
        self.qbmm_mgr.compute_rhs(self.stage_state[1], self.stage_k[1])
        test_state = 0.5 * (
            self.stage_state[0]
            + (self.stage_state[1] + self.time_step * self.stage_k[1])
        )

        # Stage 3: { y_3, k_3 } = f( t_n + 0.5 * dt, ... )
        self.stage_state[2] = 0.75 * self.stage_state[0] + 0.25 * (
            self.stage_state[1] + self.time_step * self.stage_k[1]
        )
        self.qbmm_mgr.compute_rhs(self.stage_state[2], self.stage_k[2])

        # Updates
        self.state = (
            self.stage_state[0]
            + 2.0 * (self.stage_state[2] + self.time_step * self.stage_k[2])
        ) / 3.0

        if self.adaptive:
            self.ts_error = np.linalg.norm(self.state - test_state) / np.linalg.norm(
                self.state
            )
            
        return


    def adapt_time_step_cfl(self):
        """
        This function adapts the time step for flow problems based on the CFL condition
        """
        self.time_step = self.cfl*self.domain.grid_spacing/self.domain.max_abscissa()
        return
    
            
    def adapt_time_step_RK23(self):
        """
        This function adapts the time step according to user-specified tolerance
        """

        error_fraction = np.sqrt(0.5 * self.error_tol / self.ts_error)
        time_step_factor = min(max(error_fraction, 0.3), 2.0)
        new_time_step = time_step_factor * self.time_step
        new_time_step = min(
            max(0.9 * new_time_step, self.min_time_step), self.max_time_step
        )
        self.time_step = new_time_step
        return

    def run(self):
        """
        Advancer driver
        """

        print("advancer: run: Preparing to step")

        self.time = 0.0

        self.report_step(0)
        self.write_step(0)

        i_step = 0
        step = True
        while step == True:

            self.advance()

            i_step += 1
            self.time += self.time_step

            if i_step % self.num_steps_print == 0:
                self.report_step(i_step)

            if i_step % self.num_steps_write == 0:
                self.write_step(i_step)

            if self.adaptive:
                self.adapt_time_step()

            if self.time > self.final_time or i_step >= self.num_steps:
                step = False

        print("advancer: run: stepping ends")
        print("advancer: Number of steps:", i_step)

        return

    def report_step(self, i_step):
        """
        This function reports the current state

        :param i_step: Current step
        :type i_step: int
        """

        message = (
            "advancer: step = "
            + str(i_step)
            + " ... time = "
            + "{:.16E}".format(self.time)
            + " ... time_step = "
            + "{:.16E}".format(self.time_step)
            + " ... "
        )
        print(message)
        #f_array_pretty_print(message, "state", self.state)

    def write_step(self, i_step):
        """
        This function writes the current state

        :param i_step: Current step
        :type i_step: int
        """

        message = "advancer: step = " + str(i_step) + " ... Writing to file"
        # print(message)
        self.write_to_file(i_step)

    def write_to_txt(self, i_step):
        """
        This function writes the current state to a txt file

        :param i_step: Current step
        :type i_step: int
        """
        write_flag = "a"
        if i_step == 0:
            write_flag = "w"

        with open(self.file_name, write_flag) as file_id:
            # csv.writer( file_id, delimiter=' ' ).writerow( self.state )
            csv.writer(file_id, delimiter=" ").writerow(
                [self.time] + self.state.tolist()
            )

        return

    def write_to_h5(self, i_step):
        """
        This function writes the current state to a h5 file. It is recommended for flow problems.
        """
        if i_step == 0:
            write_flag = "w"
        else:
            write_flag = "a"
        with h5py.File(self.file_name, write_flag) as f:
            g = f.create_group("step_"+str(i_step))
            g["time"] = self.time
            g["time_step"] = self.time_step
            g["state"] = self.state
        return
