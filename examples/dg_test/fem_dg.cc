/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2007 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Martin Kronbichler, Uppsala University,
 *          Wolfgang Bangerth, Texas A&M University 2007, 2008
 */


// @sect3{Include files}

// The first step, as always, is to include the functionality of these
// well-known deal.II library files and some C++ header files.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/derivative_approximation.h>
// Then we need to include some header files that provide vector, matrix, and
// preconditioner classes that implement interfaces to the respective Trilinos
// classes. In particular, we will need interfaces to the matrix and vector
// classes based on Trilinos as well as Trilinos preconditioners:
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/integrators/laplace.h>
#include <deal.II/integrators/advection.h>
#include <deal.II/integrators/l2.h>
#include <fstream>
#include <sstream>
#include <limits>


namespace fem_dg
{
using namespace dealii;
using namespace numbers;



// @sect3{Equation data}

namespace EquationData
{
const double eta = 1;
const double eta_0 = 1;
const double eta_1 = 1e+0;
const double kappa = 0;
const double beta = 1;
const double density = 1;
const double density_0 = 1.;
const double density_1 = 1.03125;
const double Max_T = 1;
const double Min_T = 0;
const double err_tol = 1e-10;


template <int dim>
class CompositionInitialValues : public Function<dim>
{
public:
    CompositionInitialValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
};


template <int dim>
double
CompositionInitialValues<dim>::value (const Point<dim> &p,
                                      const unsigned int) const
{
    return ((p(1)<=0.9)&& (p(1)>=0.7)&&(p(0)<=0.6)&&(p(0)>=0.4) ? 1:0);
}


template <int dim>
void
CompositionInitialValues<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
{
    for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = CompositionInitialValues<dim>::value (p, c);
}


template <int dim>
class CompositionRightHandSide : public Function<dim>
{
public:
    CompositionRightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
};


template <int dim>
double
CompositionRightHandSide<dim>::value (const Point<dim>  &p,
                                      const unsigned int component) const
{
    Assert (component == 0,
            ExcMessage ("Invalid operation for a scalar function."));

    Assert ((dim==2) || (dim==3), ExcNotImplemented());

    return 0;
}


template <int dim>
void
CompositionRightHandSide<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
{
    for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = CompositionRightHandSide<dim>::value (p, c);
}
}



// @sect3{Linear solvers and preconditioners}

// This section introduces some objects that are used for the solution of
// the linear equations of the Stokes system that we need to solve in each
// time step. Many of the ideas used here are the same as in step-20, where
// Schur complement based preconditioners and solvers have been introduced,
// with the actual interface taken from step-22 (in particular the
// discussion in the "Results" section of step-22, in which we introduce
// alternatives to the direct Schur complement approach). Note, however,
// that here we don't use the Schur complement to solve the Stokes
// equations, though an approximate Schur complement (the mass matrix on the
// pressure space) appears in the preconditioner.
namespace LinearSolvers
{


template <class Matrix, class Preconditioner>
class InverseMatrix : public Subscriptor
{
public:
    InverseMatrix (const Matrix         &m,
                   const Preconditioner &preconditioner);


    template <typename VectorType>
    void vmult (VectorType       &dst,
                const VectorType &src) const;

private:
    const SmartPointer<const Matrix> matrix;
    const Preconditioner &preconditioner;
};


template <class Matrix, class Preconditioner>
InverseMatrix<Matrix,Preconditioner>::
InverseMatrix (const Matrix &m,
               const Preconditioner &preconditioner)
    :
    matrix (&m),
    preconditioner (preconditioner)
{}



template <class Matrix, class Preconditioner>
template <typename VectorType>
void
InverseMatrix<Matrix,Preconditioner>::
vmult (VectorType       &dst,
       const VectorType &src) const
{
    SolverControl solver_control (src.size(), 1e-6*src.l2_norm());
    SolverCG<VectorType> cg (solver_control);

    dst = 0;

    try
    {
        cg.solve (*matrix, dst, src, preconditioner);
    }
    catch (std::exception &e)
    {
        Assert (false, ExcMessage(e.what()));
    }
}

// @sect4{Schur complement preconditioner}

// This is the implementation of the Schur complement preconditioner as
// described in detail in the introduction. As opposed to step-20 and
// step-22, we solve the block system all-at-once using GMRES, and use the
// Schur complement of the block structured matrix to build a good
// preconditioner instead.
//
// Let's have a look at the ideal preconditioner matrix
// $P=\left(\begin{array}{cc} A & 0 \\ B & -S \end{array}\right)$
// described in the introduction. If we apply this matrix in the solution
// of a linear system, convergence of an iterative GMRES solver will be
// governed by the matrix @f{eqnarray*} P^{-1}\left(\begin{array}{cc} A &
// B^T \\ B & 0 \end{array}\right) = \left(\begin{array}{cc} I & A^{-1}
// B^T \\ 0 & I \end{array}\right), @f} which indeed is very simple. A
// GMRES solver based on exact matrices would converge in one iteration,
// since all eigenvalues are equal (any Krylov method takes at most as
// many iterations as there are distinct eigenvalues). Such a
// preconditioner for the blocked Stokes system has been proposed by
// Silvester and Wathen ("Fast iterative solution of stabilised Stokes
// systems part II.  Using general block preconditioners", SIAM
// J. Numer. Anal., 31 (1994), pp. 1352-1367).
//
// Replacing <i>P</i> by $\tilde{P}$ keeps that spirit alive: the product
// $P^{-1} A$ will still be close to a matrix with eigenvalues 1 with a
// distribution that does not depend on the problem size. This lets us
// hope to be able to get a number of GMRES iterations that is
// problem-size independent.
//
// The deal.II users who have already gone through the step-20 and step-22
// tutorials can certainly imagine how we're going to implement this.  We
// replace the exact inverse matrices in $P^{-1}$ by some approximate
// inverses built from the InverseMatrix class, and the inverse Schur
// complement will be approximated by the pressure mass matrix $M_p$
// (weighted by $\eta^{-1}$ as mentioned in the introduction). As pointed
// out in the results section of step-22, we can replace the exact inverse
// of <i>A</i> by just the application of a preconditioner, in this case
// on a vector Laplace matrix as was explained in the introduction. This
// does increase the number of (outer) GMRES iterations, but is still
// significantly cheaper than an exact inverse, which would require
// between 20 and 35 CG iterations for <em>each</em> outer solver step
// (using the AMG preconditioner).
//
// Having the above explanations in mind, we define a preconditioner class
// with a <code>vmult</code> functionality, which is all we need for the
// interaction with the usual solver functions further below in the
// program code.
//
// First the declarations. These are similar to the definition of the
// Schur complement in step-20, with the difference that we need some more
// preconditioners in the constructor and that the matrices we use here
// are built upon Trilinos:
template <class PreconditionerA, class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor
{
public:
    BlockSchurPreconditioner (
        const TrilinosWrappers::BlockSparseMatrix     &S,
        const InverseMatrix<TrilinosWrappers::SparseMatrix,
        PreconditionerMp>         &Mpinv,
        const PreconditionerA                         &Apreconditioner);

    void vmult (TrilinosWrappers::BlockVector       &dst,
                const TrilinosWrappers::BlockVector &src) const;

private:
    const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> stokes_matrix;
    const SmartPointer<const InverseMatrix<TrilinosWrappers::SparseMatrix,
          PreconditionerMp > > m_inverse;
    const PreconditionerA &a_preconditioner;

    mutable TrilinosWrappers::Vector tmp;
};



template <class PreconditionerA, class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix  &S,
                         const InverseMatrix<TrilinosWrappers::SparseMatrix,
                         PreconditionerMp>      &Mpinv,
                         const PreconditionerA                      &Apreconditioner)
    :
    stokes_matrix           (&S),
    m_inverse               (&Mpinv),
    a_preconditioner        (Apreconditioner),
    tmp                     (stokes_matrix->block(1,1).m())
{}


// Next is the <code>vmult</code> function. We implement the action of
// $P^{-1}$ as described above in three successive steps.  In formulas, we
// want to compute $Y=P^{-1}X$ where $X,Y$ are both vectors with two block
// components.
//
// The first step multiplies the velocity part of the vector by a
// preconditioner of the matrix <i>A</i>, i.e. we compute $Y_0={\tilde
// A}^{-1}X_0$.  The resulting velocity vector is then multiplied by $B$
// and subtracted from the pressure, i.e. we want to compute $X_1-BY_0$.
// This second step only acts on the pressure vector and is accomplished
// by the residual function of our matrix classes, except that the sign is
// wrong. Consequently, we change the sign in the temporary pressure
// vector and finally multiply by the inverse pressure mass matrix to get
// the final pressure vector, completing our work on the Stokes
// preconditioner:
template <class PreconditionerA, class PreconditionerMp>
void
BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
vmult (TrilinosWrappers::BlockVector       &dst,
       const TrilinosWrappers::BlockVector &src) const
{
    a_preconditioner.vmult (dst.block(0), src.block(0));
    stokes_matrix->block(1,0).residual(tmp, dst.block(0), src.block(1));
    tmp *= -1;
    m_inverse->vmult (dst.block(1), tmp);
}
}



// @sect3{The <code>BoussinesqFlowProblem</code> class template}

// The definition of the class that defines the top-level logic of solving
// the time-dependent Boussinesq problem is mainly based on the step-22
// tutorial program. The main differences are that now we also have to solve
// for the composition equation, which forces us to have a second DoFHandler
// object for the composition variable as well as matrices, right hand
// sides, and solution vectors for the current and previous time steps. As
// mentioned in the introduction, all linear algebra objects are going to
// use wrappers of the corresponding Trilinos functionality.
//
// The member functions of this class are reminiscent of step-21, where we
// also used a staggered scheme that first solve the flow equations (here
// the Stokes equations, in step-21 Darcy flow) and then update the advected
// quantity (here the composition, there the saturation). The functions that
// are new are mainly concerned with determining the time step, as well as
// the proper size of the artificial viscosity stabilization.
//
// The last three variables indicate whether the various matrices or
// preconditioners need to be rebuilt the next time the corresponding build
// functions are called. This allows us to move the corresponding
// <code>if</code> into the respective function and thereby keeping our main
// <code>run()</code> function clean and easy to read.
template <int dim>
class BoussinesqFlowProblem
{
public:
    typedef MeshWorker::IntegrationInfo<dim> CellInfo;
    BoussinesqFlowProblem ();
    void run ();

private:
    void setup_dofs ();
    void setup_material_id ();
    void apply_bound_preserving_limiter ();
    void assemble_stokes_preconditioner ();
    void build_stokes_preconditioner ();
    void assemble_stokes_system ();
    void assemble_composition_matrix ();
    double get_maximal_velocity () const;
    std::pair<double,double> get_extrapolated_composition_range () const;
    void solve ();
    void output_results () const;
    void refine_mesh (const unsigned int max_grid_level);

    double
    compute_viscosity(const std::vector<double>          &old_composition,
                      const std::vector<double>          &old_old_composition,
                      const std::vector<Tensor<1,dim> >  &old_composition_grads,
                      const std::vector<Tensor<1,dim> >  &old_old_composition_grads,
                      const std::vector<double>          &old_composition_laplacians,
                      const std::vector<double>          &old_old_composition_laplacians,
                      const std::vector<Tensor<1,dim> >  &old_velocity_values,
                      const std::vector<Tensor<1,dim> >  &old_old_velocity_values,
                      const std::vector<double>          &gamma_values,
                      const double                        global_u_infty,
                      const double                        global_T_variation,
                      const double                        cell_diameter) const;


    Triangulation<dim>                  triangulation;
    const MappingQ1<dim>                mapping;
    double                              global_Omega_diameter;

    const unsigned int                  stokes_degree;
    FESystem<dim>                       stokes_fe;
    DoFHandler<dim>                     stokes_dof_handler;
    ConstraintMatrix                    stokes_constraints;

    std::vector<types::global_dof_index> stokes_block_sizes;
    TrilinosWrappers::BlockSparseMatrix stokes_matrix;
    TrilinosWrappers::BlockSparseMatrix stokes_preconditioner_matrix;

    TrilinosWrappers::BlockVector       stokes_solution;
    TrilinosWrappers::BlockVector       old_stokes_solution;
    TrilinosWrappers::BlockVector       stokes_rhs;


    const unsigned int                  composition_degree;
    FE_DGQ<dim>                         composition_fe;
    DoFHandler<dim>                     composition_dof_handler;
    ConstraintMatrix                    composition_constraints;

    TrilinosWrappers::SparseMatrix      composition_mass_matrix;
    TrilinosWrappers::SparseMatrix      composition_matrix;

    SparsityPattern                     DG_sparsity_pattern;
    SparseMatrix<double>                DG_composition_mass_matrix;
    SparseMatrix<double>                DG_composition_diff_x_matrix;
    SparseMatrix<double>                DG_composition_diff_y_matrix;
    SparseMatrix<double>                DG_composition_advec_matrix;
    SparseMatrix<double>                DG_composition_matrix;
    SparseMatrix<double>                DG_composition_diff_q1_b_matrix;
    SparseMatrix<double>                DG_composition_diff_q2_b_matrix;
    SparseDirectUMFPACK                 inverse_mass_matrix;

    TrilinosWrappers::Vector            composition_solution;
    TrilinosWrappers::Vector            old_composition_solution;
    TrilinosWrappers::Vector            old_old_composition_solution;
    TrilinosWrappers::Vector            old_old_old_composition_solution;
    TrilinosWrappers::Vector            composition_rhs;

    Vector<double>                      velocity_solution;
    Vector<double>                      DG_composition_solution;
    Vector<double>                      DG_old_composition_solution;
    Vector<double>                      DG_old_old_composition_solution;
    Vector<double>                      DG_old_old_old_composition_solution;
    Vector<double>                      DG_composition_rhs;
/*
    Vector<double>                      DG_composition_q1_solution;
    Vector<double>                      DG_old_composition_q1_solution;
    Vector<double>                      DG_old_old_composition_q1_solution;
    Vector<double>                      DG_old_old_old_composition_q1_solution;
    Vector<double>                      DG_composition_q1_rhs;

    Vector<double>                      DG_composition_q2_solution;
    Vector<double>                      DG_old_composition_q2_solution;
    Vector<double>                      DG_old_old_composition_q2_solution;
    Vector<double>                      DG_composition_q2_rhs;

*/

    double                              time;
    double                              time_step;
    double                              old_time_step;
    unsigned int                        timestep_number;

    double                              global_compositional_integrals;
    double                              local_compositional_integrals;
    std_cxx11::shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditioner;
    std_cxx11::shared_ptr<TrilinosWrappers::PreconditionIC>  Mp_preconditioner;

    bool                                rebuild_stokes_matrix;
    bool                                rebuild_composition_matrices;
    bool                                rebuild_stokes_preconditioner;

#define AMR

#define OUTPUT_FILE
#ifdef OUTPUT_FILE
    //output results
    std::ofstream output_file_m;
    std::ofstream output_file_c;
    const string output_path_m = "./output_m.txt";
    const string output_path_c = "./output_c.txt";
#endif
    // adding DG part

    typedef MeshWorker::DoFInfo<dim> DoFInfo;

    void integrate_cell_term_mass (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef);
    void integrate_cell_term_advection (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef);
    void integrate_cell_term_source (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef);
    void integrate_boundary_term_advection (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef);
    void integrate_face_term_advection (DoFInfo &dinfo1,
        DoFInfo &dinfo2,
        CellInfo &info1,
        CellInfo &info2,
        TrilinosWrappers::BlockVector &coef);
};

// @sect4{The local integrators}

// These are the functions given to the MeshWorker::integration_loop()
// called just above. They compute the local contributions to the system
// matrix and right hand side on cells and faces.
template <int dim>
void BoussinesqFlowProblem<dim>::integrate_cell_term_advection (DoFInfo &dinfo,
    CellInfo &info,
    TrilinosWrappers::BlockVector &coef)
{
  const FEValuesBase<dim> &fe_v = info.fe_values();
  FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
  const std::vector<double> &JxW = fe_v.get_JxW_values ();

  //construct stokes_cell and fe_values
  typename DoFHandler<dim>::active_cell_iterator stokes_cell(&(dinfo.cell->get_triangulation()),
      dinfo.cell->level(),
      dinfo.cell->index(),
      &stokes_dof_handler);
  const QGauss<dim> quadrature_formula(composition_degree+1);
  const unsigned int n_q_points = quadrature_formula.size();
  FEValues<dim> stokes_fe_values(stokes_fe,quadrature_formula,update_values);
  stokes_fe_values.reinit(stokes_cell);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  std::vector<double> pressure_values (n_q_points);
  std::vector<Tensor<1,dim> > velocity_values(n_q_points);

  stokes_fe_values[velocities].get_function_values (coef,
      velocity_values);
  stokes_fe_values[pressure].get_function_values (coef,
      pressure_values);

  for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
  {
    for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
    {   for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
      {
        local_matrix(i,j) -= velocity_values[point]*fe_v.shape_grad(i,point)*
          fe_v.shape_value(j,point) *
          JxW[point];
      }
    }
  }
}

template <int dim>
void BoussinesqFlowProblem<dim>::integrate_cell_term_source (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef)
{
}


template <int dim>
void BoussinesqFlowProblem<dim>::integrate_cell_term_mass (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef)
//              double time)
{
    // First, let us retrieve some of the objects used here from @p info. Note
    // that these objects can handle much more complex structures, thus the
    // access here looks more complicated than might seem necessary.
    const FEValuesBase<dim> &fe_v = info.fe_values();
    FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
    Vector<double> &local_vector = dinfo.vector(0).block(0);
    const std::vector<double> &JxW = fe_v.get_JxW_values ();
    std::vector<double> g(fe_v.n_quadrature_points);
    EquationData::CompositionRightHandSide<dim>  composition_right_hand_side;
    composition_right_hand_side.value_list (fe_v.get_quadrature_points(), g);

    typename DoFHandler<dim>::active_cell_iterator stokes_cell(&(dinfo.cell->get_triangulation()),
            dinfo.cell->level(),
            dinfo.cell->index(),
            &stokes_dof_handler);
    const QGauss<dim> quadrature_formula(composition_degree+1);
    const unsigned int n_q_points = fe_v.n_quadrature_points;
    FEValues<dim> stokes_fe_values(stokes_fe,quadrature_formula,update_values);
    stokes_fe_values.reinit(stokes_cell);

    const FEValuesExtractors::Vector velocities (0);

    std::vector<Tensor<1,dim> > velocity_values(n_q_points);

    stokes_fe_values[velocities].get_function_values (coef,
            velocity_values);
    // With these objects, we continue local integration like always. First,
    // we loop over the quadrature points and compute the advection vector in
    // the current point.
    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        {   for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                local_matrix(i,j) += fe_v.shape_value(i,point)*
                                     fe_v.shape_value(j,point) *
                                     JxW[point];

            local_vector(i) += g[point]*
                               fe_v.shape_value(i,point) *
                               JxW[point];
        }
    }
}

template <int dim>
void BoussinesqFlowProblem<dim>::integrate_boundary_term_advection (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef)
{
    const FEValuesBase<dim> &fe_v = info.fe_values();
    FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
    Vector<double> &local_vector = dinfo.vector(0).block(0);

    const std::vector<double> &JxW = fe_v.get_JxW_values ();
    const std::vector<Point<dim> > &normals = fe_v.get_normal_vectors ();

    std::vector<double> g(fe_v.n_quadrature_points);

    //BoundaryValues<dim> boundary_function(t);
    //boundary_function.value_list (fe_v.get_quadrature_points(), g);

    //construct stokes_cell and fe_values
    typename DoFHandler<dim>::active_cell_iterator stokes_cell(&(dinfo.cell->get_triangulation()),
            dinfo.cell->level(),
            dinfo.cell->index(),
            &stokes_dof_handler);
    const QGauss<dim> quadrature_formula(composition_degree+1);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> stokes_fe_values(stokes_fe,quadrature_formula,update_values);
    stokes_fe_values.reinit(stokes_cell);

    const FEValuesExtractors::Vector velocities (0);

    std::vector<Tensor<1,dim> > velocity_values(n_q_points);

    stokes_fe_values[velocities].get_function_values (coef,
            velocity_values);

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        const double beta_n=velocity_values[point]* normals[point];
        if (beta_n>0)
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
                    local_matrix(i,j) += beta_n *
                                         fe_v.shape_value(j,point) *
                                         fe_v.shape_value(i,point) *
                                         JxW[point];
        else

            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                local_vector(i) -= beta_n *
                                   g[point]*
                                   fe_v.shape_value(i,point) *
                                   JxW[point];
    }
}


template <int dim>
void BoussinesqFlowProblem<dim>::integrate_face_term_advection (DoFInfo &dinfo1,
    DoFInfo &dinfo2,
    CellInfo &info1,
    CellInfo &info2,
    TrilinosWrappers::BlockVector &coef)
{
  const FEValuesBase<dim> &fe_v = info1.fe_values();

  // For additional shape functions, we have to ask the neighbors
  // FEValuesBase.
  const FEValuesBase<dim> &fe_v_neighbor = info2.fe_values();

  FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0,false).matrix;
  FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0,true).matrix;
  FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0,true).matrix;
  FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0,false).matrix;

  const std::vector<double> &JxW = fe_v.get_JxW_values ();
  const std::vector<Point<dim> > &normals = fe_v.get_normal_vectors ();

  //construct stokes_cell and fe_values
  typename DoFHandler<dim>::active_cell_iterator stokes_cell(&(dinfo1.cell->get_triangulation()),
      dinfo1.cell->level(),
      dinfo1.cell->index(),
      &stokes_dof_handler);
  
  const QGauss<dim-1> quadrature_formula(composition_degree+1);
  const unsigned int n_q_points = quadrature_formula.size();

  FEFaceValues<dim> stokes_fe_values(stokes_fe,quadrature_formula,update_values);
  stokes_fe_values.reinit(stokes_cell,dinfo1.face_number);

  const FEValuesExtractors::Vector velocities (0);

  std::vector<Tensor<1,dim> > velocity_values(n_q_points);

  stokes_fe_values[velocities].get_function_values (coef,
      velocity_values);

  for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
  {
    const double beta_n=velocity_values[point] * normals[point];
    if (beta_n>=0)
    {
      // This term we've already seen:
      for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
          u1_v1_matrix(i,j) += beta_n *
            fe_v.shape_value(j,point) *
            fe_v.shape_value(i,point) *
            JxW[point];

      // We additionally assemble the term $(\beta\cdot n u,\hat
      // v)_{\partial \kappa_+}$,
      for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
        for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
          u1_v2_matrix(k,j) -= beta_n *
            fe_v.shape_value(j,point) *
            fe_v_neighbor.shape_value(k,point) *
            JxW[point];
    }
    else
    {
      // This one we've already seen, too:
      for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
          u2_v1_matrix(i,l) += beta_n *
            fe_v_neighbor.shape_value(l,point) *
            fe_v.shape_value(i,point) *
            JxW[point];
      // And this is another new one: $(\beta\cdot n \hat u,\hat
      // v)_{\partial \kappa_-}$:
      for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
        for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
          u2_v2_matrix(k,l) -= beta_n *
            fe_v_neighbor.shape_value(l,point) *
            fe_v_neighbor.shape_value(k,point) *
            JxW[point];
    }
  }
}


// @sect3{BoussinesqFlowProblem class implementation}

// @sect4{BoussinesqFlowProblem::BoussinesqFlowProblem}
//
// The constructor of this class is an extension of the constructor in
// step-22. We need to add the various variables that concern the
// composition. As discussed in the introduction, we are going to use
// $Q_2\times Q_1$ (Taylor-Hood) elements again for the Stokes part, and
// $Q_2$ elements for the composition. However, by using variables that
// store the polynomial degree of the Stokes and composition finite
// elements, it is easy to consistently modify the degree of the elements as
// well as all quadrature formulas used on them downstream. Moreover, we
// initialize the time stepping as well as the options for matrix assembly
// and preconditioning:
template <int dim>
BoussinesqFlowProblem<dim>::BoussinesqFlowProblem ()
    :
//    triangulation (Triangulation<dim>::maximum_smoothing),
    triangulation (),
    mapping(),
    stokes_degree (1),
    stokes_fe (FE_Q<dim>(stokes_degree+1), dim,
               FE_Q<dim>(stokes_degree), 1),
    stokes_dof_handler (triangulation),

    composition_degree (2),
    composition_fe (composition_degree),
    composition_dof_handler (triangulation),

    time (0),
    time_step (0),
    old_time_step (0),
    timestep_number (0),
    global_compositional_integrals (0),
    local_compositional_integrals (0),
    rebuild_stokes_matrix (true),
    rebuild_composition_matrices (true),
    rebuild_stokes_preconditioner (true)
{
#ifdef OUTPUT_FILE
    output_file_m.open(output_path_m.c_str());
    output_file_c.open(output_path_c.c_str());
#endif
}



// @sect4{BoussinesqFlowProblem::get_maximal_velocity}

// Starting the real functionality of this class is a helper function that
// determines the maximum ($L_\infty$) velocity in the domain (at the
// quadrature points, in fact). How it works should be relatively obvious to
// all who have gotten to this point of the tutorial. Note that since we are
// only interested in the velocity, rather than using
// <code>stokes_fe_values.get_function_values</code> to get the values of
// the entire Stokes solution (velocities and pressures) we use
// <code>stokes_fe_values[velocities].get_function_values</code> to extract
// only the velocities part. This has the additional benefit that we get it
// as a Tensor<1,dim>, rather than some components in a Vector<double>,
// allowing us to process it right away using the <code>norm()</code>
// function to get the magnitude of the velocity.
//
// The only point worth thinking about a bit is how to choose the quadrature
// points we use here. Since the goal of this function is to find the
// maximal velocity over a domain by looking at quadrature points on each
// cell. So we should ask how we should best choose these quadrature points
// on each cell. To this end, recall that if we had a single $Q_1$ field
// (rather than the vector-valued field of higher order) then the maximum
// would be attained at a vertex of the mesh. In other words, we should use
// the QTrapez class that has quadrature points only at the vertices of
// cells.
//
// For higher order shape functions, the situation is more complicated: the
// maxima and minima may be attained at points between the support points of
// shape functions (for the usual $Q_p$ elements the support points are the
// equidistant Lagrange interpolation points); furthermore, since we are
// looking for the maximum magnitude of a vector-valued quantity, we can
// even less say with certainty where the set of potential maximal points
// are. Nevertheless, intuitively if not provably, the Lagrange
// interpolation points appear to be a better choice than the Gauss points.
//
// There are now different methods to produce a quadrature formula with
// quadrature points equal to the interpolation points of the finite
// element. One option would be to use the
// FiniteElement::get_unit_support_points() function, reduce the output to a
// unique set of points to avoid duplicate function evaluations, and create
// a Quadrature object using these points. Another option, chosen here, is
// to use the QTrapez class and combine it with the QIterated class that
// repeats the QTrapez formula on a number of sub-cells in each coordinate
// direction. To cover all support points, we need to iterate it
// <code>stokes_degree+1</code> times since this is the polynomial degree of
// the Stokes element in use:
template <int dim>
double BoussinesqFlowProblem<dim>::get_maximal_velocity () const
{
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
            stokes_degree+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (stokes_fe, quadrature_formula, update_values);
    std::vector<Tensor<1,dim> > velocity_values(n_q_points);
    double max_velocity = 0;

    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    endc = stokes_dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit (cell);
        fe_values[velocities].get_function_values (stokes_solution,
                velocity_values);

        for (unsigned int q=0; q<n_q_points; ++q)
            max_velocity = std::max (max_velocity, velocity_values[q].norm());
    }

    return max_velocity;
}




// @sect4{BoussinesqFlowProblem::get_extrapolated_composition_range}

// Next a function that determines the minimum and maximum composition at
// quadrature points inside $\Omega$ when extrapolated from the two previous
// time steps to the current one. We need this information in the
// computation of the artificial viscosity parameter $\nu$ as discussed in
// the introduction.
//
// The formula for the extrapolated composition is
// $\left(1+\frac{k_n}{k_{n-1}} \right)T^{n-1} + \frac{k_n}{k_{n-1}}
// T^{n-2}$. The way to compute it is to loop over all quadrature points and
// update the maximum and minimum value if the current value is
// bigger/smaller than the previous one. We initialize the variables that
// store the max and min before the loop over all quadrature points by the
// smallest and the largest number representable as a double. Then we know
// for a fact that it is larger/smaller than the minimum/maximum and that
// the loop over all quadrature points is ultimately going to update the
// initial value with the correct one.
//
// The only other complication worth mentioning here is that in the first
// time step, $T^{k-2}$ is not yet available of course. In that case, we can
// only use $T^{k-1}$ which we have from the initial composition. As
// quadrature points, we use the same choice as in the previous function
// though with the difference that now the number of repetitions is
// determined by the polynomial degree of the composition field.
template <int dim>
std::pair<double,double>
BoussinesqFlowProblem<dim>::get_extrapolated_composition_range () const
{
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
            composition_degree);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (composition_fe, quadrature_formula,
                             update_values);
    std::vector<double> old_composition_values(n_q_points);
    std::vector<double> old_old_composition_values(n_q_points);

    if (timestep_number != 0)
    {
        double min_composition = std::numeric_limits<double>::max(),
               max_composition = -std::numeric_limits<double>::max();

        typename DoFHandler<dim>::active_cell_iterator
        cell = composition_dof_handler.begin_active(),
        endc = composition_dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            fe_values.get_function_values (old_composition_solution,
                                           old_composition_values);
            fe_values.get_function_values (old_old_composition_solution,
                                           old_old_composition_values);

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                const double composition =
                    (1. + time_step/old_time_step) * old_composition_values[q]-
                    time_step/old_time_step * old_old_composition_values[q];

                min_composition = std::min (min_composition, composition);
                max_composition = std::max (max_composition, composition);
            }
        }

        return std::make_pair(min_composition, max_composition);
    }
    else
    {
        double min_composition = std::numeric_limits<double>::max(),
               max_composition = -std::numeric_limits<double>::max();

        typename DoFHandler<dim>::active_cell_iterator
        cell = composition_dof_handler.begin_active(),
        endc = composition_dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            fe_values.get_function_values (old_composition_solution,
                                           old_composition_values);

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                const double composition = old_composition_values[q];

                min_composition = std::min (min_composition, composition);
                max_composition = std::max (max_composition, composition);
            }
        }

        return std::make_pair(min_composition, max_composition);
    }
}






// @sect4{BoussinesqFlowProblem::setup_dofs}
//
// This is the function that sets up the DoFHandler objects we have here
// (one for the Stokes part and one for the composition part) as well as set
// to the right sizes the various objects required for the linear algebra in
// this program. Its basic operations are similar to what we do in step-22.
//
// The body of the function first enumerates all degrees of freedom for the
// Stokes and composition systems. For the Stokes part, degrees of freedom
// are then sorted to ensure that velocities precede pressure DoFs so that
// we can partition the Stokes matrix into a $2\times 2$ matrix. As a
// difference to step-22, we do not perform any additional DoF
// renumbering. In that program, it paid off since our solver was heavily
// dependent on ILU's, whereas we use AMG here which is not sensitive to the
// DoF numbering. The IC preconditioner for the inversion of the pressure
// mass matrix would of course take advantage of a Cuthill-McKee like
// renumbering, but its costs are low compared to the velocity portion, so
// the additional work does not pay off.
//
// We then proceed with the generation of the hanging node constraints that
// arise from adaptive grid refinement for both DoFHandler objects. For the
// velocity, we impose no-flux boundary conditions $\mathbf{u}\cdot
// \mathbf{n}=0$ by adding constraints to the object that already stores the
// hanging node constraints matrix. The second parameter in the function
// describes the first of the velocity components in the total dof vector,
// which is zero here. The variable <code>no_normal_flux_boundaries</code>
// denotes the boundary indicators for which to set the no flux boundary
// conditions; here, this is boundary indicator zero.
//
// After having done so, we count the number of degrees of freedom in the
// various blocks:
template <int dim>
void BoussinesqFlowProblem<dim>::setup_dofs ()
{
    std::vector<unsigned int> stokes_sub_blocks (dim+1,0);
    stokes_sub_blocks[dim] = 1;

    {
        stokes_dof_handler.distribute_dofs (stokes_fe);
        DoFRenumbering::component_wise (stokes_dof_handler, stokes_sub_blocks);

        stokes_constraints.clear ();
        DoFTools::make_hanging_node_constraints (stokes_dof_handler,
                stokes_constraints);
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert (0);
        VectorTools::compute_no_normal_flux_constraints (stokes_dof_handler, 0,
                no_normal_flux_boundaries,
                stokes_constraints);
        stokes_constraints.close ();
    }
    {
        composition_dof_handler.distribute_dofs (composition_fe);

        composition_constraints.clear ();
        DoFTools::make_hanging_node_constraints (composition_dof_handler,
                composition_constraints);
        composition_constraints.close ();
    }

    std::vector<types::global_dof_index> stokes_dofs_per_block (2);
    DoFTools::count_dofs_per_block (stokes_dof_handler, stokes_dofs_per_block,
                                    stokes_sub_blocks);

    const unsigned int n_u = stokes_dofs_per_block[0],
                       n_p = stokes_dofs_per_block[1],
                       n_T = composition_dof_handler.n_dofs();

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << " (on "
              << triangulation.n_levels()
              << " levels)"
              << std::endl
              << "Number of degrees of freedom: "
              << n_u + n_p + n_T
              << " (" << n_u << '+' << n_p << '+'<< n_T <<')'
              << std::endl
              << std::endl;

    // The next step is to create the sparsity pattern for the Stokes and
    // composition system matrices as well as the preconditioner matrix from
    // which we build the Stokes preconditioner. As in step-22, we choose to
    // create the pattern not as in the first few tutorial programs, but by
    // using the blocked version of CompressedSimpleSparsityPattern.  The
    // reason for doing this is mainly memory, that is, the SparsityPattern
    // class would consume too much memory when used in three spatial
    // dimensions as we intend to do for this program.
    //
    // So, we first release the memory stored in the matrices, then set up an
    // object of type BlockCompressedSimpleSparsityPattern consisting of
    // $2\times 2$ blocks (for the Stokes system matrix and preconditioner) or
    // CompressedSimpleSparsityPattern (for the composition part). We then
    // fill these objects with the nonzero pattern, taking into account that
    // for the Stokes system matrix, there are no entries in the
    // pressure-pressure block (but all velocity vector components couple with
    // each other and with the pressure). Similarly, in the Stokes
    // preconditioner matrix, only the diagonal blocks are nonzero, since we
    // use the vector Laplacian as discussed in the introduction. This
    // operator only couples each vector component of the Laplacian with
    // itself, but not with the other vector components. (Application of the
    // constraints resulting from the no-flux boundary conditions will couple
    // vector components at the boundary again, however.)
    //
    // When generating the sparsity pattern, we directly apply the constraints
    // from hanging nodes and no-flux boundary conditions. This approach was
    // already used in step-27, but is different from the one in early
    // tutorial programs where we first built the original sparsity pattern
    // and only then added the entries resulting from constraints. The reason
    // for doing so is that later during assembly we are going to distribute
    // the constraints immediately when transferring local to global
    // dofs. Consequently, there will be no data written at positions of
    // constrained degrees of freedom, so we can let the
    // DoFTools::make_sparsity_pattern function omit these entries by setting
    // the last Boolean flag to <code>false</code>. Once the sparsity pattern
    // is ready, we can use it to initialize the Trilinos matrices. Since the
    // Trilinos matrices store the sparsity pattern internally, there is no
    // need to keep the sparsity pattern around after the initialization of
    // the matrix.
    stokes_block_sizes.resize (2);
    stokes_block_sizes[0] = n_u;
    stokes_block_sizes[1] = n_p;
    {
        stokes_matrix.clear ();

        BlockCompressedSimpleSparsityPattern csp (2,2);

        csp.block(0,0).reinit (n_u, n_u);
        csp.block(0,1).reinit (n_u, n_p);
        csp.block(1,0).reinit (n_p, n_u);
        csp.block(1,1).reinit (n_p, n_p);

        csp.collect_sizes ();

        Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);

        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (! ((c==dim) && (d==dim)))
                    coupling[c][d] = DoFTools::always;
                else
                    coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern (stokes_dof_handler, coupling, csp,
                                         stokes_constraints, false);

        stokes_matrix.reinit (csp);
    }

    {
        Amg_preconditioner.reset ();
        Mp_preconditioner.reset ();
        stokes_preconditioner_matrix.clear ();

        BlockCompressedSimpleSparsityPattern csp (2,2);

        csp.block(0,0).reinit (n_u, n_u);
        csp.block(0,1).reinit (n_u, n_p);
        csp.block(1,0).reinit (n_p, n_u);
        csp.block(1,1).reinit (n_p, n_p);

        csp.collect_sizes ();

        Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c == d)
                    coupling[c][d] = DoFTools::always;
                else
                    coupling[c][d] = DoFTools::none;

        DoFTools::make_sparsity_pattern (stokes_dof_handler, coupling, csp,
                                         stokes_constraints, false);

        stokes_preconditioner_matrix.reinit (csp);
    }

    // The creation of the composition matrix (or, rather, matrices, since we
    // provide a composition mass matrix and a composition stiffness matrix,
    // that will be added together for time discretization) follows the
    // generation of the Stokes matrix &ndash; except that it is much easier
    // here since we do not need to take care of any blocks or coupling
    // between components. Note how we initialize the three composition
    // matrices: We only use the sparsity pattern for reinitialization of the
    // first matrix, whereas we use the previously generated matrix for the
    // two remaining reinits. The reason for doing so is that reinitialization
    // from an already generated matrix allows Trilinos to reuse the sparsity
    // pattern instead of generating a new one for each copy. This saves both
    // some time and memory.
    {
        composition_mass_matrix.clear ();
        composition_matrix.clear ();

        CompressedSparsityPattern csp(composition_dof_handler.n_dofs());
        DoFTools::make_flux_sparsity_pattern (composition_dof_handler, csp);
        DG_sparsity_pattern.copy_from(csp);

        DG_composition_matrix.reinit (DG_sparsity_pattern);
        DG_composition_mass_matrix.reinit (DG_sparsity_pattern);
        DG_composition_advec_matrix.reinit (DG_sparsity_pattern);
    }

    // Lastly, we set the vectors for the Stokes solutions $\mathbf u^{n-1}$
    // and $\mathbf u^{n-2}$, as well as for the compositions $T^{n}$,
    // $T^{n-1}$ and $T^{n-2}$ (required for time stepping) and all the system
    // right hand sides to their correct sizes and block structure:
    stokes_solution.reinit (stokes_block_sizes);
    old_stokes_solution.reinit (stokes_block_sizes);
    stokes_rhs.reinit (stokes_block_sizes);

    velocity_solution.reinit (composition_dof_handler.n_dofs());
    DG_composition_solution.reinit (composition_dof_handler.n_dofs());
    DG_old_composition_solution.reinit (composition_dof_handler.n_dofs());
    DG_old_old_composition_solution.reinit (composition_dof_handler.n_dofs());
    DG_old_old_old_composition_solution.reinit (composition_dof_handler.n_dofs());

    DG_composition_rhs.reinit (composition_dof_handler.n_dofs());

    composition_solution=DG_composition_solution;
    old_composition_solution=DG_old_composition_solution;
    old_old_composition_solution=DG_old_old_composition_solution;
    old_old_old_composition_solution=DG_old_old_old_composition_solution;
    composition_rhs=DG_composition_rhs;
}


// @sect4{BoussinesqFlowProblem::setup_material_id}
//
template <int dim>
void BoussinesqFlowProblem<dim>::setup_material_id ()
{

    typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    endc = stokes_dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        const Point<dim> cell_center = cell->center();
        if (   (cell_center(0)<=.625*1)
                && (cell_center(0)>=.375*1)
                && (cell_center(1)<=.875*1)
                && (cell_center(1)>=.625*1)
           )
            cell->set_material_id(1);
        else
            cell->set_material_id(0);
    }

}
// @sect4{BoussinesqFlowProblem::apply_bound_preserving_limiter}
//
template <int dim>
void BoussinesqFlowProblem<dim>::apply_bound_preserving_limiter ()
{

    typename DoFHandler<dim>::active_cell_iterator
    cell = composition_dof_handler.begin_active(),
    endc = composition_dof_handler.end();

    std::cout << "Stat to apply Bound Preserving Limiter "
              << std::endl;
#if 0
    const QGauss<dim-1> quadrature_formula_x(composition_degree+1);
    const QGauss<dim-1> quadrature_formula_y(composition_degree+1);
#else
    const QGauss<dim-1> quadrature_formula_x(composition_degree+1);
    const QGaussLobatto<dim-1> quadrature_formula_y(composition_degree+1);
#endif
    
    const QGauss<dim> quadrature_formula(composition_degree+1);
    const QAnisotropic<dim> quadrature_formula_xy(quadrature_formula_x,quadrature_formula_y);
    const QAnisotropic<dim> quadrature_formula_yx(quadrature_formula_y,quadrature_formula_x);

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_q_points_xy = quadrature_formula_xy.size();
    const unsigned int n_q_points_yx = quadrature_formula_yx.size();
    
    FEValues<dim>  composition_fe_values (composition_fe, quadrature_formula,
                                          update_values    |
                                          update_gradients |
                                          update_hessians  |
                                          update_quadrature_points  |
                                          update_JxW_values);
    std::vector<double>   composition_values(n_q_points);
    
    FEValues<dim>  composition_fe_values_xy (composition_fe, quadrature_formula_xy,
                                          update_values    |
                                          update_gradients |
                                          update_hessians  |
                                          update_quadrature_points  |
                                          update_JxW_values);
    std::vector<double>   composition_values_xy (n_q_points_xy);

    FEValues<dim>  composition_fe_values_yx (composition_fe, quadrature_formula_yx,
                                          update_values    |
                                          update_gradients |
                                          update_hessians  |
                                          update_quadrature_points  |
                                          update_JxW_values);
    std::vector<double>   composition_values_yx (n_q_points_yx);

    const unsigned int dofs_per_cell   = composition_fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    global_compositional_integrals=0;
    local_compositional_integrals=0;
    for (; cell!=endc; ++cell)
    {
        cell->get_dof_indices (local_dof_indices);
        double min_composition = DG_composition_solution(local_dof_indices[0]),
               max_composition = DG_composition_solution(local_dof_indices[0]);
        for (unsigned int i=1; i<local_dof_indices.size(); ++i)
        {
            min_composition = std::min<double> (min_composition,
                                                DG_composition_solution(local_dof_indices[i]));
            max_composition = std::max<double> (max_composition,
                                                DG_composition_solution(local_dof_indices[i]));
        }
        //Find the trouble cell
        if(
            (max_composition>=(EquationData::Max_T+EquationData::err_tol))
            ||
            (min_composition<=(EquationData::Min_T-EquationData::err_tol))
        )
        {

            //Find the average of solution on this trouble cell
            double T_cell_average=0;
            double T_cell_area=0;

            composition_fe_values_xy.reinit (cell);
            composition_fe_values_xy.get_function_values (DG_composition_solution,
                    composition_values_xy);
            for (unsigned int q=0; q<n_q_points_xy; ++q)
            {
              T_cell_average+= composition_values_xy[q]
                *composition_fe_values_xy.JxW(q);
              T_cell_area+= 1.0
                *composition_fe_values_xy.JxW(q);
              min_composition = std::min<double> (min_composition,
                  composition_values_xy[q]);
              max_composition = std::max<double> (max_composition,
                  composition_values_xy[q]);
            }
            
            // Need to divide the area of the cell
            T_cell_average/=T_cell_area;
            
            for (unsigned int q=0; q<n_q_points_yx; ++q)
            {
              min_composition = std::min<double> (min_composition,
                  composition_values_yx[q]);
              max_composition = std::max<double> (max_composition,
                  composition_values_yx[q]);
            }
            
            //Define theta
            double theta_T=std::min<double>(1,abs((EquationData::Max_T-T_cell_average)/(max_composition-T_cell_average)));
            theta_T=std::min<double>(theta_T,abs((EquationData::Min_T-T_cell_average)/(min_composition-T_cell_average)));
            //Modify the numerical solution at quadrature points
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                double t_tmp=DG_composition_solution(local_dof_indices[i]);
                DG_composition_solution(local_dof_indices[i])=theta_T*(t_tmp-T_cell_average)+T_cell_average;
            }
            /*
            //check the cell average again
            T_cell_average=0;

            for (unsigned int q=0; q<n_q_points_xy; ++q)
            {
            T_cell_average+= composition_values_xy[q]
             *composition_fe_values_xy.JxW(q);
             }
            // Need to divide the area of the cell
            T_cell_average/=T_cell_area;
            */

        }
        composition_solution=DG_composition_solution;
#if 0 
        const Point<dim> cell_center = cell->center();
        if (   (cell_center(0)<=.675)
            && (cell_center(0)>=.325)
            && (cell_center(1)<=.9)
            && (cell_center(1)>=.00001)
           )
        {
#if 1
          composition_fe_values.reinit (cell);
          composition_fe_values.get_function_values (DG_composition_solution,
              composition_values);
          local_compositional_integrals=0;
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            local_compositional_integrals+= composition_values[q]
              *composition_fe_values.JxW(q);
          }
          global_compositional_integrals+=local_compositional_integrals;
#else   
          composition_fe_values_xy.reinit (cell);
          composition_fe_values_xy.get_function_values (DG_composition_solution,
              composition_values_xy);
          local_compositional_integrals=0;
          for (unsigned int q=0; q<n_q_points_xy; ++q)
          {
            local_compositional_integrals+= composition_values_xy[q]
              *composition_fe_values_xy.JxW(q);
          }
          global_compositional_integrals+=local_compositional_integrals;
#endif
        }

    }

    std::cout << "total compositional mass is \n"
      << global_compositional_integrals
      << std::endl;
#endif
}



// @sect4{BoussinesqFlowProblem::assemble_stokes_preconditioner}
//
// This function assembles the matrix we use for preconditioning the Stokes
// system. What we need are a vector Laplace matrix on the velocity
// components and a mass matrix weighted by $\eta^{-1}$ on the pressure
// component. We start by generating a quadrature object of appropriate
// order, the FEValues object that can give values and gradients at the
// quadrature points (together with quadrature weights). Next we create data
// structures for the cell matrix and the relation between local and global
// DoFs. The vectors <code>grad_phi_u</code> and <code>phi_p</code> are
// going to hold the values of the basis functions in order to faster build
// up the local matrices, as was already done in step-22. Before we start
// the loop over all active cells, we have to specify which components are
// pressure and which are velocity.
template <int dim>
void
BoussinesqFlowProblem<dim>::assemble_stokes_preconditioner ()
{
    stokes_preconditioner_matrix = 0;

    const QGauss<dim> quadrature_formula(stokes_degree+2);
    FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
                                        update_JxW_values |
                                        update_values |
                                        update_gradients);

    const unsigned int   dofs_per_cell   = stokes_fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    FEValues<dim>     composition_fe_values (composition_fe, quadrature_formula,
            update_values);
    std::vector<double>               old_composition_values(n_q_points);

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Tensor<2,dim> > grad_phi_u (dofs_per_cell);
    std::vector<double>         phi_p      (dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    endc = stokes_dof_handler.end();
    typename DoFHandler<dim>::active_cell_iterator
    composition_cell = composition_dof_handler.begin_active();

    for (; cell!=endc; ++cell, ++composition_cell)
    {
        stokes_fe_values.reinit (cell);
        composition_fe_values.reinit (composition_cell);
        local_matrix = 0;

        // The creation of the local matrix is rather simple. There are only a
        // Laplace term (on the velocity) and a mass matrix weighted by
        // $\eta^{-1}$ to be generated, so the creation of the local matrix is
        // done in two lines. Once the local matrix is ready (loop over rows
        // and columns in the local matrix on each quadrature point), we get
        // the local DoF indices and write the local information into the
        // global matrix. We do this as in step-27, i.e. we directly apply the
        // constraints from hanging nodes locally. By doing so, we don't have
        // to do that afterwards, and we don't also write into entries of the
        // matrix that will actually be set to zero again later when
        // eliminating constraints.
        composition_fe_values.get_function_values (old_composition_solution,
                old_composition_values);
        for (unsigned int q=0; q<n_q_points; ++q)
        {
            const double old_composition = old_composition_values[q];
            for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
                grad_phi_u[k] = stokes_fe_values[velocities].gradient(k,q);
                phi_p[k]      = stokes_fe_values[pressure].value (k, q);
            }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    local_matrix(i,j) += ((EquationData::eta_0
                                           +(EquationData::eta_1-EquationData::eta_0)*old_composition) *
                                          scalar_product (grad_phi_u[i], grad_phi_u[j])
                                          +
                                          (1./(EquationData::eta_0
                                               +(EquationData::eta_1-EquationData::eta_0)*old_composition)) *
                                          phi_p[i] * phi_p[j])
                                         * stokes_fe_values.JxW(q);
        }

        cell->get_dof_indices (local_dof_indices);
        stokes_constraints.distribute_local_to_global (local_matrix,
                local_dof_indices,
                stokes_preconditioner_matrix);
    }
}



// @sect4{BoussinesqFlowProblem::build_stokes_preconditioner}
//
// This function generates the inner preconditioners that are going to be
// used for the Schur complement block preconditioner. Since the
// preconditioners need only to be regenerated when the matrices change,
// this function does not have to do anything in case the matrices have not
// changed (i.e., the flag <code>rebuild_stokes_preconditioner</code> has
// the value <code>false</code>). Otherwise its first task is to call
// <code>assemble_stokes_preconditioner</code> to generate the
// preconditioner matrices.
//
// Next, we set up the preconditioner for the velocity-velocity matrix
// <i>A</i>. As explained in the introduction, we are going to use an AMG
// preconditioner based on a vector Laplace matrix $\hat{A}$ (which is
// spectrally close to the Stokes matrix <i>A</i>). Usually, the
// TrilinosWrappers::PreconditionAMG class can be seen as a good black-box
// preconditioner which does not need any special knowledge. In this case,
// however, we have to be careful: since we build an AMG for a vector
// problem, we have to tell the preconditioner setup which dofs belong to
// which vector component. We do this using the function
// DoFTools::extract_constant_modes, a function that generates a set of
// <code>dim</code> vectors, where each one has ones in the respective
// component of the vector problem and zeros elsewhere. Hence, these are the
// constant modes on each component, which explains the name of the
// variable.
template <int dim>
void
BoussinesqFlowProblem<dim>::build_stokes_preconditioner ()
{
    if (rebuild_stokes_preconditioner == false)
        return;

    std::cout << "   Rebuilding Stokes preconditioner..." << std::flush;

    assemble_stokes_preconditioner ();

    Amg_preconditioner = std_cxx11::shared_ptr<TrilinosWrappers::PreconditionAMG>
                         (new TrilinosWrappers::PreconditionAMG());

    std::vector<std::vector<bool> > constant_modes;
    FEValuesExtractors::Vector velocity_components(0);
    DoFTools::extract_constant_modes (stokes_dof_handler,
                                      stokes_fe.component_mask(velocity_components),
                                      constant_modes);
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.constant_modes = constant_modes;

    // Next, we set some more options of the AMG preconditioner. In
    // particular, we need to tell the AMG setup that we use quadratic basis
    // functions for the velocity matrix (this implies more nonzero elements
    // in the matrix, so that a more robust algorithm needs to be chosen
    // internally). Moreover, we want to be able to control how the coarsening
    // structure is build up. The way the Trilinos smoothed aggregation AMG
    // does this is to look which matrix entries are of similar size as the
    // diagonal entry in order to algebraically build a coarse-grid
    // structure. By setting the parameter <code>aggregation_threshold</code>
    // to 0.02, we specify that all entries that are more than two percent of
    // size of some diagonal pivots in that row should form one coarse grid
    // point. This parameter is rather ad hoc, and some fine-tuning of it can
    // influence the performance of the preconditioner. As a rule of thumb,
    // larger values of <code>aggregation_threshold</code> will decrease the
    // number of iterations, but increase the costs per iteration. A look at
    // the Trilinos documentation will provide more information on these
    // parameters. With this data set, we then initialize the preconditioner
    // with the matrix we want it to apply to.
    //
    // Finally, we also initialize the preconditioner for the inversion of the
    // pressure mass matrix. This matrix is symmetric and well-behaved, so we
    // can chose a simple preconditioner. We stick with an incomplete Cholesky
    // (IC) factorization preconditioner, which is designed for symmetric
    // matrices. We could have also chosen an SSOR preconditioner with
    // relaxation factor around 1.2, but IC is cheaper for our example. We
    // wrap the preconditioners into a <code>std_cxx11::shared_ptr</code>
    // pointer, which makes it easier to recreate the preconditioner next time
    // around since we do not have to care about destroying the previously
    // used object.
    amg_data.elliptic = true;
    amg_data.higher_order_elements = true;
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 0.02;
    Amg_preconditioner->initialize(stokes_preconditioner_matrix.block(0,0),
                                   amg_data);

    Mp_preconditioner = std_cxx11::shared_ptr<TrilinosWrappers::PreconditionIC>
                        (new TrilinosWrappers::PreconditionIC());
    Mp_preconditioner->initialize(stokes_preconditioner_matrix.block(1,1));

    std::cout << std::endl;

    rebuild_stokes_preconditioner = false;
}



// @sect4{BoussinesqFlowProblem::assemble_stokes_system}
//
// The time lag scheme we use for advancing the coupled Stokes-composition
// system forces us to split up the assembly (and the solution of linear
// systems) into two step. The first one is to create the Stokes system
// matrix and right hand side, and the second is to create matrix and right
// hand sides for the composition dofs, which depends on the result of the
// linear system for the velocity.
//
// This function is called at the beginning of each time step. In the first
// time step or if the mesh has changed, indicated by the
// <code>rebuild_stokes_matrix</code>, we need to assemble the Stokes
// matrix; on the other hand, if the mesh hasn't changed and the matrix is
// already available, this is not necessary and all we need to do is
// assemble the right hand side vector which changes in each time step.
//
// Regarding the technical details of implementation, not much has changed
// from step-22. We reset matrix and vector, create a quadrature formula on
// the cells, and then create the respective FEValues object. For the update
// flags, we require basis function derivatives only in case of a full
// assembly, since they are not needed for the right hand side; as always,
// choosing the minimal set of flags depending on what is currently needed
// makes the call to FEValues::reinit further down in the program more
// efficient.
//
// There is one thing that needs to be commented &ndash; since we have a
// separate finite element and DoFHandler for the composition, we need to
// generate a second FEValues object for the proper evaluation of the
// composition solution. This isn't too complicated to realize here: just
// use the composition structures and set an update flag for the basis
// function values which we need for evaluation of the composition
// solution. The only important part to remember here is that the same
// quadrature formula is used for both FEValues objects to ensure that we
// get matching information when we loop over the quadrature points of the
// two objects.
//
// The declarations proceed with some shortcuts for array sizes, the
// creation of the local matrix and right hand side as well as the vector
// for the indices of the local dofs compared to the global system.
template <int dim>
void BoussinesqFlowProblem<dim>::assemble_stokes_system ()
{
  std::cout << "   Assembling..." << std::flush;

  if (rebuild_stokes_matrix == true)
    stokes_matrix=0;

  stokes_rhs=0;

  const QGauss<dim> quadrature_formula (stokes_degree+2);
  FEValues<dim>     stokes_fe_values (stokes_fe, quadrature_formula,
      update_values    |
      update_quadrature_points  |
      update_JxW_values |
      (rebuild_stokes_matrix == true
       ?
       update_gradients
       :
       UpdateFlags(0)));

  FEValues<dim> composition_fe_values (composition_fe, quadrature_formula,
      update_values);

  const unsigned int   dofs_per_cell   = stokes_fe.dofs_per_cell;
  const unsigned int   n_q_points      = quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs    (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  // Next we need a vector that will contain the values of the composition
  // solution at the previous time level at the quadrature points to
  // assemble the source term in the right hand side of the momentum
  // equation. Let's call this vector <code>old_solution_values</code>.
  //
  // The set of vectors we create next hold the evaluations of the basis
  // functions as well as their gradients and symmetrized gradients that
  // will be used for creating the matrices. Putting these into their own
  // arrays rather than asking the FEValues object for this information each
  // time it is needed is an optimization to accelerate the assembly
  // process, see step-22 for details.
  //
  // The last two declarations are used to extract the individual blocks
  // (velocity, pressure, composition) from the total FE system.
  std::vector<double>               old_composition_values(n_q_points);

  std::vector<Tensor<1,dim> >          phi_u       (dofs_per_cell);
  std::vector<SymmetricTensor<2,dim> > grads_phi_u (dofs_per_cell);
  std::vector<double>                  div_phi_u   (dofs_per_cell);
  std::vector<double>                  phi_p       (dofs_per_cell);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  // Now start the loop over all cells in the problem. We are working on two
  // different DoFHandlers for this assembly routine, so we must have two
  // different cell iterators for the two objects in use. This might seem a
  // bit peculiar, since both the Stokes system and the composition system
  // use the same grid, but that's the only way to keep degrees of freedom
  // in sync. The first statements within the loop are again all very
  // familiar, doing the update of the finite element data as specified by
  // the update flags, zeroing out the local arrays and getting the values
  // of the old solution at the quadrature points. Then we are ready to loop
  // over the quadrature points on the cell.
  typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
         endc = stokes_dof_handler.end();
  typename DoFHandler<dim>::active_cell_iterator
    composition_cell = composition_dof_handler.begin_active();

  for (; cell!=endc; ++cell, ++composition_cell)
  {
    stokes_fe_values.reinit (cell);
    composition_fe_values.reinit (composition_cell);

    local_matrix = 0;
    local_rhs = 0;

    composition_fe_values.get_function_values (old_composition_solution,
        old_composition_values);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
      const double old_composition = old_composition_values[q];

      // Next we extract the values and gradients of basis functions
      // relevant to the terms in the inner products. As shown in
      // step-22 this helps accelerate assembly.
      //
      // Once this is done, we start the loop over the rows and columns
      // of the local matrix and feed the matrix with the relevant
      // products. The right hand side is filled with the forcing term
      // driven by composition in direction of gravity (which is
      // vertical in our example).  Note that the right hand side term
      // is always generated, whereas the matrix contributions are only
      // updated when it is requested by the
      // <code>rebuild_matrices</code> flag.
      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
        phi_u[k] = stokes_fe_values[velocities].value (k,q);
        if (rebuild_stokes_matrix)
        {
          grads_phi_u[k] = stokes_fe_values[velocities].symmetric_gradient(k,q);
          div_phi_u[k]   = stokes_fe_values[velocities].divergence (k, q);
          phi_p[k]       = stokes_fe_values[pressure].value (k, q);
        }
      }

      if (rebuild_stokes_matrix)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            local_matrix(i,j) += ((EquationData::eta_0
                  +(EquationData::eta_1-EquationData::eta_0)*old_composition) *
                2* (grads_phi_u[i] * grads_phi_u[j])
                - div_phi_u[i] * phi_p[j]
                - phi_p[i] * div_phi_u[j])
              * stokes_fe_values.JxW(q);
      const Point<dim> gravity = -( (dim == 2) ? (Point<dim> (0,1)) :
          (Point<dim> (0,0,1)) );
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        local_rhs(i) += (-(EquationData::density_0
              +(EquationData::density_1-EquationData::density_0)*old_composition) *
            EquationData::beta *
            gravity * phi_u[i]) *
          stokes_fe_values.JxW(q);
    }

    // The last step in the loop over all cells is to enter the local
    // contributions into the global matrix and vector structures to the
    // positions specified in <code>local_dof_indices</code>.  Again, we
    // let the ConstraintMatrix class do the insertion of the cell matrix
    // elements to the global matrix, which already condenses the hanging
    // node constraints.
    cell->get_dof_indices (local_dof_indices);

    if (rebuild_stokes_matrix == true)
      stokes_constraints.distribute_local_to_global (local_matrix,
          local_rhs,
          local_dof_indices,
          stokes_matrix,
          stokes_rhs);
    else
      stokes_constraints.distribute_local_to_global (local_rhs,
          local_dof_indices,
          stokes_rhs);
  }

  rebuild_stokes_matrix = false;

  std::cout << std::endl;
}




// @sect4{BoussinesqFlowProblem::assemble_composition_matrix}
//
// This function assembles the matrix in the composition equation. The
// composition matrix consists of two parts, a mass matrix and the time step
// size times a stiffness matrix given by a Laplace term times the amount of
// diffusion. Since the matrix depends on the time step size (which varies
// from one step to another), the composition matrix needs to be updated
// every time step. We could simply regenerate the matrices in every time
// step, but this is not really efficient since mass and Laplace matrix do
// only change when we change the mesh. Hence, we do this more efficiently
// by generating two separate matrices in this function, one for the mass
// matrix and one for the stiffness (diffusion) matrix. We will then sum up
// the matrix plus the stiffness matrix times the time step size once we
// know the actual time step.
//
// So the details for this first step are very simple. In case we need to
// rebuild the matrix (i.e., the mesh has changed), we zero the data
// structures, get a quadrature formula and a FEValues object, and create
//local matrices, local dof indices and evaluation structures for the basis
// functions.
template <int dim>
void BoussinesqFlowProblem<dim>::assemble_composition_matrix ()
{

    DG_composition_mass_matrix = 0;

    MeshWorker::IntegrationInfoBox<dim> info_box;

    const unsigned int n_gauss_points = composition_dof_handler.get_fe().degree+1;
    info_box.initialize_gauss_quadrature(n_gauss_points,
                                         n_gauss_points,
                                         n_gauss_points);

    info_box.initialize_update_flags();
    UpdateFlags update_flags = update_quadrature_points |
                               update_values            |
                               update_gradients;
    info_box.add_update_flags(update_flags, true, true, true, true);

    info_box.initialize(composition_fe, mapping);

    Vector<double> rhs_tmp (composition_dof_handler.n_dofs());

    MeshWorker::DoFInfo<dim> dof_info(composition_dof_handler);


    DG_composition_mass_matrix.reinit (DG_sparsity_pattern);
    MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;
    assembler.initialize(DG_composition_mass_matrix,rhs_tmp);

    auto integrate_cell_term_mass_bind=std::bind(&BoussinesqFlowProblem<dim>::integrate_cell_term_mass,this, std::placeholders::_1, std::placeholders::_2, this->stokes_solution);
    MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
      (composition_dof_handler.begin_active(), composition_dof_handler.end(),
       dof_info, info_box,
       integrate_cell_term_mass_bind,
       NULL,
       NULL,
       assembler);

    //DG_composition_rhs=rhs_tmp;//if there is a source term;//FIXME
    //copy the DG mass matrix to Trillinor mass matrix
    composition_mass_matrix.reinit(DG_composition_mass_matrix);

    // obtain advection matrix
    DG_composition_advec_matrix.reinit (DG_sparsity_pattern);
    rhs_tmp.reinit(composition_dof_handler.n_dofs());
    MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> > assembler1;
    assembler1.initialize(DG_composition_advec_matrix, rhs_tmp);

    //bind definitions
    auto integrate_cell_term_advection_bind = std::bind(&BoussinesqFlowProblem<dim>::integrate_cell_term_advection,
            this, std::placeholders::_1, std::placeholders::_2, this->stokes_solution);
    auto integrate_boundary_term_advection_bind = std::bind(&BoussinesqFlowProblem<dim>::integrate_boundary_term_advection,
            this, std::placeholders::_1, std::placeholders::_2, this->stokes_solution);
    auto integrate_face_term_advection_bind = std::bind(&BoussinesqFlowProblem<dim>::integrate_face_term_advection,
            this, std::placeholders::_1, std::placeholders::_2,std::placeholders::_3, std::placeholders::_4, this->stokes_solution);

    MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
    (composition_dof_handler.begin_active(), composition_dof_handler.end(),
     dof_info, info_box,
     integrate_cell_term_advection_bind,
     integrate_boundary_term_advection_bind,
     integrate_face_term_advection_bind,
     assembler1);

    composition_mass_matrix.reinit(DG_composition_mass_matrix);
}



// @sect4{BoussinesqFlowProblem::solve}
//
// This function solves the linear systems of equations. Following the
// introduction, we start with the Stokes system, where we need to generate
// our block Schur preconditioner. Since all the relevant actions are
// implemented in the class <code>BlockSchurPreconditioner</code>, all we
// have to do is to initialize the class appropriately. What we need to pass
// down is an <code>InverseMatrix</code> object for the pressure mass
// matrix, which we set up using the respective class together with the IC
// preconditioner we already generated, and the AMG preconditioner for the
// velocity-velocity matrix. Note that both <code>Mp_preconditioner</code>
// and <code>Amg_preconditioner</code> are only pointers, so we use
// <code>*</code> to pass down the actual preconditioner objects.
//
// Once the preconditioner is ready, we create a GMRES solver for the block
// system. Since we are working with Trilinos data structures, we have to
// set the respective template argument in the solver. GMRES needs to
// internally store temporary vectors for each iteration (see the discussion
// in the results section of step-22) &ndash; the more vectors it can use,
// the better it will generally perform. To keep memory demands in check, we
// set the number of vectors to 100. This means that up to 100 solver
// iterations, every temporary vector can be stored. If the solver needs to
// iterate more often to get the specified tolerance, it will work on a
// reduced set of vectors by restarting at every 100 iterations.
//
// With this all set up, we solve the system and distribute the constraints
// in the Stokes system, i.e. hanging nodes and no-flux boundary condition,
// in order to have the appropriate solution values even at constrained
// dofs. Finally, we write the number of iterations to the screen.
template <int dim>
void BoussinesqFlowProblem<dim>::solve ()
{
    std::cout << "   Solving..." << std::endl;

    {
        const LinearSolvers::InverseMatrix<TrilinosWrappers::SparseMatrix,
              TrilinosWrappers::PreconditionIC>
              mp_inverse (stokes_preconditioner_matrix.block(1,1), *Mp_preconditioner);

        const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
              TrilinosWrappers::PreconditionIC>
              preconditioner (stokes_matrix, mp_inverse, *Amg_preconditioner);

        SolverControl solver_control (stokes_matrix.m(),
                                      1e-6*stokes_rhs.l2_norm());

        SolverGMRES<TrilinosWrappers::BlockVector>
        gmres (solver_control,
               SolverGMRES<TrilinosWrappers::BlockVector >::AdditionalData(100));

        for (unsigned int i=0; i<stokes_solution.size(); ++i)
            if (stokes_constraints.is_constrained(i))
                stokes_solution(i) = 0;

        stokes_rhs*=-1.0;
        gmres.solve(stokes_matrix, stokes_solution, stokes_rhs, preconditioner);

        stokes_constraints.distribute (stokes_solution);

        std::cout << "   "
                  << solver_control.last_step()
                  << " GMRES iterations for Stokes subsystem."
                  << std::endl;
    }
    // Once we know the Stokes solution, we can determine the new time step

    old_time_step = time_step;
    const double maximal_velocity = get_maximal_velocity();

#if 1
    if (maximal_velocity >= 0.0001)
        //time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
        //          composition_degree *
        time_step = 0.05*GridTools::minimal_cell_diameter(triangulation) /
                    maximal_velocity;
    else
        //time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
        //         composition_degree *
        time_step = 0.05*GridTools::minimal_cell_diameter(triangulation) /
                    .0001;
#else
    time_step = 2;
#endif
    std::cout << "   " << "Time step: " << time_step
              << std::endl;


    // Next we set up the composition system and the right hand side using the
    {
	if (timestep_number<=1)
	{// SSP RK Stage 1
    Vector<double> sol_tmp (composition_dof_handler.n_dofs());
		DG_composition_advec_matrix.vmult(DG_composition_rhs,DG_old_composition_solution);
		DG_composition_rhs*=-time_step;
		DG_composition_mass_matrix.vmult(sol_tmp,DG_old_composition_solution);
		DG_composition_rhs+=sol_tmp;
		SolverControl solver_control (DG_composition_mass_matrix.m(),
				1e-8*DG_composition_rhs.l2_norm());
		SolverCG<> cg (solver_control);

		PreconditionBlockSSOR<SparseMatrix<double> > preconditioner;
		preconditioner.initialize(DG_composition_mass_matrix, composition_fe.dofs_per_cell);

		cg.solve (DG_composition_mass_matrix, DG_composition_solution,
				DG_composition_rhs, preconditioner);

		std::cout << "   "
			<< solver_control.last_step()
			<< " CG iterations 1 for composition."
			<< std::endl;
		apply_bound_preserving_limiter();
		// SSP RK Stage 2
		DG_composition_advec_matrix.vmult(DG_composition_rhs,DG_composition_solution);
		DG_composition_rhs*=-time_step*0.5;
		DG_old_composition_solution+=DG_composition_solution;
		DG_old_composition_solution*=0.5;
		DG_composition_mass_matrix.vmult(sol_tmp,DG_old_composition_solution);
		DG_composition_rhs+=sol_tmp;

		preconditioner.initialize(DG_composition_mass_matrix, composition_fe.dofs_per_cell);
		cg.solve (DG_composition_mass_matrix, DG_composition_solution,
				DG_composition_rhs, preconditioner);

		std::cout << "   "
			<< solver_control.last_step()
			<< " CG iterations 2 for composition."
			<< std::endl;
	}
	else
	{	// SSP 2nd multistep method
		DG_composition_advec_matrix.vmult(DG_composition_rhs,DG_old_composition_solution);
		DG_composition_rhs*=-time_step*1.5;
    
    Vector<double> rhs_tmp (composition_dof_handler.n_dofs());
	  
    DG_composition_mass_matrix.vmult(rhs_tmp,DG_old_old_old_composition_solution);
		rhs_tmp*=0.25;
		DG_composition_rhs+=rhs_tmp;
		DG_composition_mass_matrix.vmult(rhs_tmp,DG_old_composition_solution);
		rhs_tmp*=0.75;
		DG_composition_rhs+=rhs_tmp;

		SolverControl solver_control (DG_composition_mass_matrix.m(),
				1e-8*DG_composition_rhs.l2_norm());
		SolverCG<> cg (solver_control);
		PreconditionBlockSSOR<SparseMatrix<double> > preconditioner;
		preconditioner.initialize(DG_composition_mass_matrix, composition_fe.dofs_per_cell);
		cg.solve (DG_composition_mass_matrix, DG_composition_solution,
				DG_composition_rhs, preconditioner);

		std::cout << "   "
			<< solver_control.last_step()
			<< " CG iterations SSP stage 2 for composition."
			<< std::endl;
	}

        composition_solution=DG_composition_solution;

    }
}



// @sect4{BoussinesqFlowProblem::output_results}
//
// This function writes the solution to a VTK output file for visualization,
// which is done every tenth time step. This is usually quite a simple task,
// since the deal.II library provides functions that do almost all the job
// for us. There is one new function compared to previous examples: We want
// to visualize both the Stokes solution and the composition as one data
// set, but we have done all the calculations based on two different
// DoFHandler objects. Luckily, the DataOut class is prepared to deal with
// it. All we have to do is to not attach one single DoFHandler at the
// beginning and then use that for all added vector, but specify the
// DoFHandler to each vector separately. The rest is done as in step-22. We
// create solution names (that are going to appear in the visualization
// program for the individual components). The first <code>dim</code>
// components are the vector velocity, and then we have pressure for the
// Stokes part, whereas composition is scalar. This information is read out
// using the DataComponentInterpretation helper class. Next, we actually
// attach the data vectors with their DoFHandler objects, build patches
// according to the degree of freedom, which are (sub-) elements that
// describe the data for visualization programs. Finally, we set a file name
// (that includes the time step number) and write the vtk file.
template <int dim>
void BoussinesqFlowProblem<dim>::output_results ()  const
{
    if (timestep_number % 10 != 0)
        return;

    std::vector<std::string> stokes_names (dim, "velocity");
    stokes_names.push_back ("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    stokes_component_interpretation
    (dim+1, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i=0; i<dim; ++i)
        stokes_component_interpretation[i]
            = DataComponentInterpretation::component_is_part_of_vector;

    DataOut<dim> data_out;
    data_out.add_data_vector (stokes_dof_handler, stokes_solution,
                              stokes_names, stokes_component_interpretation);
    data_out.add_data_vector (composition_dof_handler, composition_solution,
                              "C");
    data_out.add_data_vector (composition_dof_handler, DG_composition_solution,
                              "DG_C");
    data_out.build_patches (std::min(stokes_degree, composition_degree));

    std::ostringstream filename;
    filename << "solution-" << Utilities::int_to_string(timestep_number, 4) << ".vtk";
    //filename << "solution-" << Utilities::int_to_string(timestep_number, 4) << ".gnuplot";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
    //data_out.write_gnuplot (output);
}



// @sect4{BoussinesqFlowProblem::refine_mesh}
//
// This function takes care of the adaptive mesh refinement. The three tasks
// this function performs is to first find out which cells to
// refine/coarsen, then to actually do the refinement and eventually
// transfer the solution vectors between the two different grids. The first
// task is simply achieved by using the well-established Kelly error
// estimator on the composition (it is the composition we're mainly
// interested in for this program, and we need to be accurate in regions of
// high composition gradients, also to not have too much numerical
// diffusion). The second task is to actually do the remeshing. That
// involves only basic functions as well, such as the
// <code>refine_and_coarsen_fixed_fraction</code> that refines those cells
// with the largest estimated error that together make up 80 per cent of the
// error, and coarsens those cells with the smallest error that make up for
// a combined 10 per cent of the error.
//
// If implemented like this, we would get a program that will not make much
// progress: Remember that we expect composition fields that are nearly
// discontinuous (the diffusivity $\kappa$ is very small after all) and
// consequently we can expect that a freely adapted mesh will refine further
// and further into the areas of large gradients. This decrease in mesh size
// will then be accompanied by a decrease in time step, requiring an
// exceedingly large number of time steps to solve to a given final time. It
// will also lead to meshes that are much better at resolving
// discontinuities after several mesh refinement cycles than in the
// beginning.
//
// In particular to prevent the decrease in time step size and the
// correspondingly large number of time steps, we limit the maximal
// refinement depth of the mesh. To this end, after the refinement indicator
// has been applied to the cells, we simply loop over all cells on the
// finest level and unselect them from refinement if they would result in
// too high a mesh level.
template <int dim>
void BoussinesqFlowProblem<dim>::refine_mesh (const unsigned int max_grid_level)
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
#if 1
    KellyErrorEstimator<dim>::estimate (composition_dof_handler,
                                        QGauss<dim-1>(composition_degree+1),
                                        typename FunctionMap<dim>::type(),
                                        composition_solution,
                                        estimated_error_per_cell);

    //GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
      //      estimated_error_per_cell,
        //    1, 0);
    GridRefinement::refine(triangulation,
            estimated_error_per_cell,
            .1);

#endif
    if (triangulation.n_levels() > max_grid_level)
        for (typename Triangulation<dim>::active_cell_iterator
                cell = triangulation.begin_active(max_grid_level);
                cell != triangulation.end(); ++cell)
            cell->clear_refine_flag ();

    // As part of mesh refinement we need to transfer the solution vectors
    // from the old mesh to the new one. To this end we use the
    // SolutionTransfer class and we have to prepare the solution vectors that
    // should be transferred to the new grid (we will lose the old grid once
    // we have done the refinement so the transfer has to happen concurrently
    // with refinement). What we definitely need are the current and the old
    // composition (BDF-2 time stepping requires two old solutions). Since the
    // SolutionTransfer objects only support to transfer one object per dof
    // handler, we need to collect the two composition solutions in one data
    // structure. Moreover, we choose to transfer the Stokes solution, too,
    // since we need the velocity at two previous time steps, of which only
    // one is calculated on the fly.
    //
    // Consequently, we initialize two SolutionTransfer objects for the Stokes
    // and composition DoFHandler objects, by attaching them to the old dof
    // handlers. With this at place, we can prepare the triangulation and the
    // data vectors for refinement (in this order).
    std::vector<TrilinosWrappers::Vector> x_composition (3);
    x_composition[0] = composition_solution;
    x_composition[1] = old_composition_solution;
    x_composition[2] = old_old_composition_solution;
    TrilinosWrappers::BlockVector x_stokes = stokes_solution;

    SolutionTransfer<dim,TrilinosWrappers::Vector>
    composition_trans(composition_dof_handler);
    SolutionTransfer<dim,TrilinosWrappers::BlockVector>
    stokes_trans(stokes_dof_handler);

    triangulation.prepare_coarsening_and_refinement();
    composition_trans.prepare_for_coarsening_and_refinement(x_composition);
    stokes_trans.prepare_for_coarsening_and_refinement(x_stokes);

    // Now everything is ready, so do the refinement and recreate the dof
    // structure on the new grid, and initialize the matrix structures and the
    // new vectors in the <code>setup_dofs</code> function. Next, we actually
    // perform the interpolation of the solutions between the grids. We create
    // another copy of temporary vectors for composition (now corresponding to
    // the new grid), and let the interpolate function do the job. Then, the
    // resulting array of vectors is written into the respective vector member
    // variables. For the Stokes vector, everything is just the same &ndash;
    // except that we do not need another temporary vector since we just
    // interpolate a single vector. In the end, we have to tell the program
    // that the matrices and preconditioners need to be regenerated, since the
    // mesh has changed.
    triangulation.execute_coarsening_and_refinement ();
    setup_dofs ();

    std::vector<TrilinosWrappers::Vector> tmp (3);
    tmp[0].reinit (composition_solution);
    tmp[1].reinit (composition_solution);
    tmp[2].reinit (composition_solution);
    composition_trans.interpolate(x_composition, tmp);

    composition_solution = tmp[0];
    old_composition_solution = tmp[1];
    old_old_composition_solution = tmp[2];

    DG_composition_solution = tmp[0];
    DG_old_composition_solution = tmp[1];
    DG_old_old_composition_solution = tmp[2];

    stokes_trans.interpolate (x_stokes, stokes_solution);

    rebuild_stokes_matrix         = true;
    rebuild_composition_matrices  = true;
    rebuild_stokes_preconditioner = true;
}



// @sect4{BoussinesqFlowProblem::run}
//
// This function performs all the essential steps in the Boussinesq
// program. It starts by setting up a grid (depending on the spatial
// dimension, we choose some different level of initial refinement and
// additional adaptive refinement steps, and then create a cube in
// <code>dim</code> dimensions and set up the dofs for the first time. Since
// we want to start the time stepping already with an adaptively refined
// grid, we perform some pre-refinement steps, consisting of all assembly,
// solution and refinement, but without actually advancing in time. Rather,
// we use the vilified <code>goto</code> statement to jump out of the time
// loop right after mesh refinement to start all over again on the new mesh
// beginning at the <code>start_time_iteration</code> label.
//
// Before we start, we project the initial values to the grid and obtain the
// first data for the <code>old_composition_solution</code> vector. Then, we
// initialize time step number and time step and start the time loop.
template <int dim>
void BoussinesqFlowProblem<dim>::run ()
{
    const unsigned int initial_refinement = (dim == 2 ? 4 : 2);
    const unsigned int n_pre_refinement_steps = (dim == 2 ? 5 : 3);


    GridGenerator::hyper_cube (triangulation,
                               0.,
                               1.);
    global_Omega_diameter = GridTools::diameter (triangulation);
    triangulation.refine_global (initial_refinement);
    //triangulation.refine_global (8); //if want to use global uniform mesh
    
    setup_dofs();
    setup_material_id();
    unsigned int pre_refinement_step = 0;

start_time_iteration:

    const QIterated<dim> quadrature_formula (QTrapez<1>(),
            composition_degree);

    FEValues<dim> fe_values (composition_fe, quadrature_formula,
                             update_values | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell   = composition_fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    std::vector<double>  local_vector(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = composition_dof_handler.begin_active(),
    endc = composition_dof_handler.end();

    EquationData::CompositionInitialValues<dim> composition_intial;

    for (; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int q=0; q<n_q_points; ++q)
        {
            local_vector[q]=cell->material_id();
            //local_vector[q]=composition_intial.value(fe_values.quadrature_point(q),0);
        }
        old_composition_solution.set(local_dof_indices,local_vector);
    }
    DG_old_composition_solution=old_composition_solution;
    DG_old_old_composition_solution=old_composition_solution;
    DG_composition_solution=old_composition_solution;

    timestep_number           = 0;
    time_step = old_time_step = 0;

    time =0;
    output_results ();
    do
    {
        std::cout << "Timestep " << timestep_number
                  << ":  t=" << time
                  << std::endl;

        // The first steps in the time loop are all obvious &ndash; we
        // assemble the Stokes system, the preconditioner, the composition
        // matrix (matrices and preconditioner do actually only change in case
        // we've remeshed before), and then do the solve. Before going on with
        // the next time step, we have to check whether we should first finish
        // the pre-refinement steps or if we should remesh (every fifth time
        // step), refining up to a level that is consistent with initial
        // refinement and pre-refinement steps. Last in the loop is to advance
        // the solutions, i.e. to copy the solutions to the next "older" time
        // level.
        assemble_stokes_system ();
        build_stokes_preconditioner ();
        assemble_composition_matrix ();

        solve ();
        apply_bound_preserving_limiter();
        double min_composition = DG_composition_solution(0),
               max_composition = DG_composition_solution(0);
        for (unsigned int i=0; i<DG_composition_solution.size(); ++i)
        {
            min_composition = std::min<double> (min_composition,
                                                DG_composition_solution(i));
            max_composition = std::max<double> (max_composition,
                                                DG_composition_solution(i));
        }

        std::cout << "   Composition range1: "
                  << min_composition << ' ' << max_composition
                  << std::endl;

#ifdef AMR
        if ((timestep_number == 0) &&
                (pre_refinement_step < n_pre_refinement_steps))
        {
            refine_mesh (initial_refinement + n_pre_refinement_steps);
            apply_bound_preserving_limiter();
            ++pre_refinement_step;
            timestep_number=pre_refinement_step*100+1;
            output_results ();
            timestep_number =0;
            goto start_time_iteration;
        }
        else if ((timestep_number > 0) && (timestep_number % 1 == 0))
        {
            refine_mesh (initial_refinement + n_pre_refinement_steps);
            apply_bound_preserving_limiter();
        }

#endif

        std::cout << "   Composition mass integral: "
                  << global_compositional_integrals
                  << std::endl;

        time += time_step;
        ++timestep_number;
        output_results ();

        old_stokes_solution             = stokes_solution;
        old_old_old_composition_solution = old_old_composition_solution;
        old_old_composition_solution    = old_composition_solution;
        old_composition_solution     = composition_solution;
        
        DG_old_old_old_composition_solution = DG_old_old_composition_solution;
        DG_old_old_composition_solution    = DG_old_composition_solution;
        DG_old_composition_solution     = DG_composition_solution;

        rebuild_stokes_matrix         = true;
        rebuild_composition_matrices  = true;
        rebuild_stokes_preconditioner = true;

        min_composition = DG_composition_solution(0);
        max_composition = DG_composition_solution(0);
        for (unsigned int i=0; i<DG_composition_solution.size(); ++i)
        {
            min_composition = std::min<double> (min_composition,
                                                DG_composition_solution(i));
            max_composition = std::max<double> (max_composition,
                                                DG_composition_solution(i));
        }

        std::cout << "   Composition range2: "
                  << min_composition << ' ' << max_composition
                  << std::endl;
        std::cout << "   Composition mass integral: "
                  << global_compositional_integrals
                  << std::endl;
#ifdef OUTPUT_FILE
        output_file_m << global_compositional_integrals << std::endl;
        output_file_c << min_composition << ' ' << max_composition
                      << std::endl;
#endif
    }
    // Do all the above until we arrive at time 100.
    while (timestep_number <= 5001);
}
}



// @sect3{The <code>main</code> function}
//
// The main function looks almost the same as in all other programs.
//
// There is one difference we have to be careful about. This program uses
// Trilinos and, typically, Trilinos is configured so that it can run in
// %parallel using MPI. This doesn't mean that it <i>has</i> to run in
// %parallel, and in fact this program (unlike step-32) makes no attempt at
// all to do anything in %parallel using MPI. Nevertheless, Trilinos wants the
// MPI system to be initialized. We do that be creating an object of type
// Utilities::MPI::MPI_InitFinalize that initializes MPI (if available) using
// the arguments given to main() (i.e., <code>argc</code> and
// <code>argv</code>) and de-initializes it again when the object goes out of
// scope.
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace fem_dg;

        deallog.depth_console (0);

        Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
                numbers::invalid_unsigned_int);

        BoussinesqFlowProblem<2> flow_problem;
        flow_problem.run ();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
