import numpy as np 
import utils

def compute_rhs(self, moments, rhs):
    """
    This function computes moment-transport RHS

    :param moments: Transported moments
    :param rhs: Moments rate-of-change

    """
    # Compute abscissas and weights from moments
    if self.num_internal_coords == 1:
        abscissas, weights = self.moment_invert(moments)
    else:
        abscissas, weights = self.moment_invert(moments, self.indices)

    # Loop over moments
    for i_moment in range(self.num_moments):
        # Evalue RHS terms
        if self.num_internal_coords == 1:
            exponents = [
                np.double(self.exponents[j, 0](self.indices[i_moment]))
                for j in range(self.num_exponents)
            ]
            coefficients = [
                np.double(self.coefficients[j](self.indices[i_moment]))
                for j in range(self.num_coefficients)
            ]
        elif self.num_internal_coords == 2:
            exponents = [
                [
                    np.double(
                        self.exponents[j, 0](
                            self.indices[i_moment][0], self.indices[i_moment][1]
                        )
                    ),
                    np.double(
                        self.exponents[j, 1](
                            self.indices[i_moment][0], self.indices[i_moment][1]
                        )
                    ),
                ]
                for j in range(self.num_exponents)
            ]
            coefficients = [
                np.double(
                    self.coefficients[j](
                        self.indices[i_moment][0], self.indices[i_moment][1]
                    )
                )
                for j in range(self.num_coefficients)
            ]
        else:
            print(
                "num_internal_coords", self.num_internal_coords, "not supported yet"
            )
            quit()

        # Put them in numpy arrays
        np_exponents = np.array(exponents)
        np_coefficients = np.array(coefficients)
        # Project back to moments
        rhs_moments = self.projection(weights, abscissas, np_exponents)
        # Compute RHS
        rhs[i_moment] = np.dot(np_coefficients, rhs_moments)
    #
    projected_moments = self.projection(weights, abscissas, self.indices)
    for i_moment in range(self.num_moments):
        moments[i_moment] = projected_moments[i_moment]
    #
    return

if __name__ == "__main__":
    