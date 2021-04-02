

# For 3D CHyqmom27 with flow

* nothing happens with or without projection now
* this is because the RHS is always zero
* need to find out why this is
* possible cause:
  * go to domain.compute_rhs
  * does self.grid_inversion actually return valid abscissas/weights in the important regions (jet regions)
  * if so:
    * why are all the fluxes zeros?
