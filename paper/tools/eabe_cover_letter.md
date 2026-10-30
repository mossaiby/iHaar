[Date]

The Editor
*Engineering Analysis with Boundary Elements*

Dear Editor,

Please consider the enclosed manuscript, **"Fast solvers on arbitrary grids for Haar wavelet collocation:
sharp-interface phase-field computations and adaptive elastostatics"**, for publication in *Engineering
Analysis with Boundary Elements*.

The Haar wavelet collocation method (HWCM) is widely used in engineering because it is simple to derive and
to program. In two and three dimensions it is usually solved as one dense system, which limits it to about
64 × 64 cells, and graded grids must be designed by hand. The manuscript removes these limitations:

- **Tridiagonal structure.** Written in nodal values, HWCM is midpoint quadratic-spline collocation. An
  exact three-term identity replaces the dense HWCM integration matrices by two tridiagonal matrices.
- **A solver that always applies.** We prove that the resulting second-derivative operator has real,
  simple and nonpositive eigenvalues on *every* strictly increasing grid. Fast diagonalization, which reduces
  tensor-grid problems to a few dense matrix products, is therefore well defined on arbitrary graded and
  adaptively refined grids. Its floating-point behaviour is tested separately on random and adversarial
  grids, with an inexpensive safeguard.
- **Engineering applications.** Mass-conserving Allen–Cahn and Cahn–Hilliard solvers that resolve interfaces
  with about four nodes across the transition layer, and adaptive plane elastostatics with traction
  boundaries, where refinement driven by the Haar detail coefficients recovers the optimal convergence rate
  at corner singularities.
- **Comparison with finite elements.** On identical grids, the method needed about a quarter of the unknowns
  of biquadratic finite elements. For the finite elements we used a sparse direct solver, conjugate
  gradients with a fast-diagonalization preconditioner, and conjugate gradients with algebraic multigrid.

We believe the work fits the journal's scope in collocation and mesh-reduction methods for engineering
analysis. It offers the large HWCM community a fast and provably well-defined solver for problems at
realistic resolution, and it states the limits of the method plainly: rectangular domains and compressible
materials. Truly local refinement is shown to be compatible with the fast solver through nested composite
patches, demonstrated for a scalar corner singularity.

All results are reproducible. The implementation, the tests and the scripts that produce every table and
figure are openly available at https://github.com/mossaiby/iHaar (release v1.2).

The manuscript is original, has not been published, and is not under consideration elsewhere. I am the sole
author and declare no competing interests.

[Suggested reviewers, if requested by the submission system: name, affiliation, e-mail, expertise.]

Sincerely,

Farshid Mossaiby
Department of Civil Engineering, University of Isfahan, Isfahan 81744-73441, Iran
mossaiby@eng.ui.ac.ir
