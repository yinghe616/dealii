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
#include <deal.II/lac/precondition.h>

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
#include <deal.II/numerics/matrix_tools.h>
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




// @sect3{The <code>DriftDiffusionProblem</code> class template}

// The definition of the class that defines the top-level logic of solving
// the time-dependent Boussinesq problem is mainly based on the step-22
// tutorial program. The main differences are that now we also have to solve
template <int dim>
class DriftDiffusionProblem
{
public:
    typedef MeshWorker::IntegrationInfo<dim> CellInfo;
    DriftDiffusionProblem ();
    void run ();

private:
    void setup_dofs ();
    void setup_material_id ();
    void apply_bound_preserving_limiter ();
    void assemble_poisson_preconditioner ();
    void build_poisson_preconditioner ();
    void assemble_poisson_system ();
    void assemble_concentration_matrix ();
    double get_maximal_potential () const;
    std::pair<double,double> get_extrapolated_concentration_range () const;
    void solve ();
    void output_results () const;
    void refine_mesh (const unsigned int max_grid_level);

    double
    compute_viscosity(const std::vector<double>          &old_concentration,
                      const std::vector<double>          &old_old_concentration,
                      const std::vector<Tensor<1,dim> >  &old_concentration_grads,
                      const std::vector<Tensor<1,dim> >  &old_old_concentration_grads,
                      const std::vector<double>          &old_concentration_laplacians,
                      const std::vector<double>          &old_old_concentration_laplacians,
                      const std::vector<Tensor<1,dim> >  &old_potential_values,
                      const std::vector<Tensor<1,dim> >  &old_old_potential_values,
                      const std::vector<double>          &gamma_values,
                      const double                        global_u_infty,
                      const double                        global_T_variation,
                      const double                        cell_diameter) const;


    Triangulation<dim>                  triangulation;
    const MappingQ1<dim>                mapping;
    double                              global_Omega_diameter;

    const unsigned int                  potential_degree;
    FE_Q<dim>                           potential_fe;
    DoFHandler<dim>                     potential_dof_handler;
    ConstraintMatrix                    poisson_constraints;
    SparsityPattern                     poisson_sparsity_pattern;

    SparseMatrix<double>                poisson_matrix;
    SparseMatrix<double>                poisson_preconditioner_matrix;

    Vector<double>                      poisson_rhs;
    Vector<double>                      potential_solution;
    Vector<double>                      old_potential_solution;


    const unsigned int                  concentration_degree;
    FE_DGQ<dim>                         concentration_fe;
    DoFHandler<dim>                     concentration_dof_handler;
    ConstraintMatrix                    concentration_constraints;

    TrilinosWrappers::SparseMatrix      concentration_mass_matrix;
    TrilinosWrappers::SparseMatrix      concentration_matrix;

    SparsityPattern                     DG_sparsity_pattern;
    SparseMatrix<double>                DG_concentration_mass_matrix;
    SparseMatrix<double>                DG_concentration_diff_x_matrix;
    SparseMatrix<double>                DG_concentration_diff_y_matrix;
    SparseMatrix<double>                DG_concentration_advec_matrix;
    SparseMatrix<double>                DG_concentration_matrix;
    SparseMatrix<double>                DG_concentration_diff_q1_b_matrix;
    SparseMatrix<double>                DG_concentration_diff_q2_b_matrix;
    SparseDirectUMFPACK                 inverse_mass_matrix;

    TrilinosWrappers::Vector            concentration_solution;
    TrilinosWrappers::Vector            old_concentration_solution;
    TrilinosWrappers::Vector            old_old_concentration_solution;
    TrilinosWrappers::Vector            old_old_old_concentration_solution;
    TrilinosWrappers::Vector            concentration_rhs;

    Vector<double>                      DG_concentration_solution;
    Vector<double>                      DG_old_concentration_solution;
    Vector<double>                      DG_old_old_concentration_solution;
    Vector<double>                      DG_old_old_old_concentration_solution;
    Vector<double>                      DG_concentration_rhs;
    
    Vector<double>                      DG_concentration_q1_solution;
    Vector<double>                      DG_old_concentration_q1_solution;
    Vector<double>                      DG_old_old_concentration_q1_solution;
    Vector<double>                      DG_old_old_old_concentration_q1_solution;
    Vector<double>                      DG_concentration_q1_rhs;

    Vector<double>                      DG_concentration_q2_solution;
    Vector<double>                      DG_old_concentration_q2_solution;
    Vector<double>                      DG_old_old_concentration_q2_solution;
    Vector<double>                      DG_concentration_q2_rhs;

   

    double                              time;
    double                              time_step;
    double                              old_time_step;
    unsigned int                        timestep_number;

    double                              global_concentrational_integrals;
    double                              local_concentrational_integrals;
    std_cxx11::shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditioner;
    std_cxx11::shared_ptr<TrilinosWrappers::PreconditionIC>  Mp_preconditioner;

    bool                                rebuild_poisson_matrix;
    bool                                rebuild_concentration_matrices;
    bool                                rebuild_poisson_preconditioner;

//#define AMR

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
void DriftDiffusionProblem<dim>::integrate_cell_term_advection (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef)
{
    const FEValuesBase<dim> &fe_v = info.fe_values();
    FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
    const std::vector<double> &JxW = fe_v.get_JxW_values ();

    //construct poisson_cell and fe_values
    typename DoFHandler<dim>::active_cell_iterator poisson_cell(&(dinfo.cell->get_triangulation()),
            dinfo.cell->level(),
            dinfo.cell->index(),
            &potential_dof_handler);
    const QGauss<dim> quadrature_formula(concentration_degree+1);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> potential_fe_values(potential_fe,quadrature_formula,update_values);
    potential_fe_values.reinit(poisson_cell);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    std::vector<double> pressure_values (n_q_points);
    std::vector<Tensor<1,dim> > potential_values(n_q_points);

    potential_fe_values.get_function_values (coef,
            potential_values);

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        {   for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
            {
//                local_matrix(i,j) -= potential_values*fe_v.shape_grad(i,point)*
                                     fe_v.shape_value(j,point) *
                                     JxW[point];
            }
        }
    }
}

template <int dim>
void DriftDiffusionProblem<dim>::integrate_cell_term_source (DoFInfo &dinfo,
        CellInfo &info,
        TrilinosWrappers::BlockVector &coef)
{
}


template <int dim>
void DriftDiffusionProblem<dim>::integrate_cell_term_mass (DoFInfo &dinfo,
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
    EquationData::CompositionRightHandSide<dim>  concentration_right_hand_side;
    concentration_right_hand_side.value_list (fe_v.get_quadrature_points(), g);

    typename DoFHandler<dim>::active_cell_iterator poisson_cell(&(dinfo.cell->get_triangulation()),
            dinfo.cell->level(),
            dinfo.cell->index(),
            &potential_dof_handler);
    const QGauss<dim> quadrature_formula(concentration_degree+1);
    const unsigned int n_q_points = fe_v.n_quadrature_points;
    FEValues<dim> potential_fe_values(potential_fe,quadrature_formula,update_values);
    potential_fe_values.reinit(poisson_cell);

    const FEValuesExtractors::Vector velocities (0);

    std::vector<Tensor<1,dim> > potential_values(n_q_points);

    potential_fe_values[velocities].get_function_values (coef,
            potential_values);
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
void DriftDiffusionProblem<dim>::integrate_boundary_term_advection (DoFInfo &dinfo,
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

    //construct poisson_cell and fe_values
    typename DoFHandler<dim>::active_cell_iterator poisson_cell(&(dinfo.cell->get_triangulation()),
            dinfo.cell->level(),
            dinfo.cell->index(),
            &potential_dof_handler);
    const QGauss<dim> quadrature_formula(concentration_degree+1);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> potential_fe_values(potential_fe,quadrature_formula,update_values);
    potential_fe_values.reinit(poisson_cell);

    const FEValuesExtractors::Vector velocities (0);

    std::vector<Tensor<1,dim> > potential_values(n_q_points);

    potential_fe_values[velocities].get_function_values (coef,
            potential_values);

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        const double beta_n=potential_values[point]* normals[point];
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
void DriftDiffusionProblem<dim>::integrate_face_term_advection (DoFInfo &dinfo1,
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

    //construct poisson_cell and fe_values
    typename DoFHandler<dim>::active_cell_iterator poisson_cell(&(dinfo1.cell->get_triangulation()),
            dinfo1.cell->level(),
            dinfo1.cell->index(),
            &potential_dof_handler);

    const QGauss<dim-1> quadrature_formula(concentration_degree+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEFaceValues<dim> potential_fe_values(potential_fe,quadrature_formula,update_values);
    potential_fe_values.reinit(poisson_cell,dinfo1.face_number);

    const FEValuesExtractors::Vector velocities (0);

    std::vector<Tensor<1,dim> > potential_values(n_q_points);

    potential_fe_values[velocities].get_function_values (coef,
            potential_values);

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        const double beta_n=potential_values[point] * normals[point];
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


// @sect3{DriftDiffusionProblem class implementation}

// @sect4{DriftDiffusionProblem::DriftDiffusionProblem}
//
// The constructor of this class is an extension of the constructor in
// step-22. We need to add the various variables that concern the
// concentration. As discussed in the introduction, we are going to use
// $Q_2\times Q_1$ (Taylor-Hood) elements again for the Stokes part, and
// $Q_2$ elements for the concentration. However, by using variables that
// store the polynomial degree of the Stokes and concentration finite
// elements, it is easy to consistently modify the degree of the elements as
// well as all quadrature formulas used on them downstream. Moreover, we
// initialize the time stepping as well as the options for matrix assembly
// and preconditioning:
template <int dim>
DriftDiffusionProblem<dim>::DriftDiffusionProblem ()
    :
//    triangulation (Triangulation<dim>::maximum_smoothing),
    triangulation (),
    mapping(),
    potential_degree (2),
    potential_fe (potential_degree),
    potential_dof_handler (triangulation),

    concentration_degree (2),
    concentration_fe (concentration_degree),
    concentration_dof_handler (triangulation),

    time (0),
    time_step (0),
    old_time_step (0),
    timestep_number (0),
    global_concentrational_integrals (0),
    local_concentrational_integrals (0),
    rebuild_poisson_matrix (true),
    rebuild_concentration_matrices (true),
    rebuild_poisson_preconditioner (true)
{
#ifdef OUTPUT_FILE
    output_file_m.open(output_path_m.c_str());
    output_file_c.open(output_path_c.c_str());
#endif
}



// @sect4{DriftDiffusionProblem::get_maximal_potential}

template <int dim>
double DriftDiffusionProblem<dim>::get_maximal_potential () const
{
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
            potential_degree+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (potential_fe, quadrature_formula, update_values);
    std::vector<double > potential_values(n_q_points);
    double max_potential = 0;

    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
    cell = potential_dof_handler.begin_active(),
    endc = potential_dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        fe_values.reinit (cell);
        fe_values.get_function_values (potential_solution,
                potential_values);

        for (unsigned int q=0; q<n_q_points; ++q)
            max_potential = std::max (max_potential, std::abs(potential_values[q]));
    }

    return max_potential;
}




// @sect4{DriftDiffusionProblem::get_extrapolated_concentration_range}

template <int dim>
std::pair<double,double>
DriftDiffusionProblem<dim>::get_extrapolated_concentration_range () const
{
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
            concentration_degree);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (concentration_fe, quadrature_formula,
                             update_values);
    std::vector<double> old_concentration_values(n_q_points);
    std::vector<double> old_old_concentration_values(n_q_points);

    if (timestep_number != 0)
    {
        double min_concentration = std::numeric_limits<double>::max(),
               max_concentration = -std::numeric_limits<double>::max();

        typename DoFHandler<dim>::active_cell_iterator
        cell = concentration_dof_handler.begin_active(),
        endc = concentration_dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            fe_values.get_function_values (old_concentration_solution,
                                           old_concentration_values);
            fe_values.get_function_values (old_old_concentration_solution,
                                           old_old_concentration_values);

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                const double concentration =
                    (1. + time_step/old_time_step) * old_concentration_values[q]-
                    time_step/old_time_step * old_old_concentration_values[q];

                min_concentration = std::min (min_concentration, concentration);
                max_concentration = std::max (max_concentration, concentration);
            }
        }

        return std::make_pair(min_concentration, max_concentration);
    }
    else
    {
        double min_concentration = std::numeric_limits<double>::max(),
               max_concentration = -std::numeric_limits<double>::max();

        typename DoFHandler<dim>::active_cell_iterator
        cell = concentration_dof_handler.begin_active(),
        endc = concentration_dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            fe_values.reinit (cell);
            fe_values.get_function_values (old_concentration_solution,
                                           old_concentration_values);

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                const double concentration = old_concentration_values[q];

                min_concentration = std::min (min_concentration, concentration);
                max_concentration = std::max (max_concentration, concentration);
            }
        }

        return std::make_pair(min_concentration, max_concentration);
    }
}






// @sect4{DriftDiffusionProblem::setup_dofs}
//
template <int dim>
void DriftDiffusionProblem<dim>::setup_dofs ()
{

    {
        potential_dof_handler.distribute_dofs (potential_fe);
        poisson_constraints.clear ();
        DoFTools::make_hanging_node_constraints (potential_dof_handler,
                poisson_constraints);
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert (0);
        VectorTools::compute_no_normal_flux_constraints (potential_dof_handler, 0,
                no_normal_flux_boundaries,
                poisson_constraints);
        poisson_constraints.close ();
    }
    
    {
        concentration_dof_handler.distribute_dofs (concentration_fe);

        concentration_constraints.clear ();
        DoFTools::make_hanging_node_constraints (concentration_dof_handler,
                concentration_constraints);
        concentration_constraints.close ();
    }


    const unsigned int n_u = potential_dof_handler.n_dofs(),
                       n_T = concentration_dof_handler.n_dofs();

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << " (on "
              << triangulation.n_levels()
              << " levels)"
              << std::endl
              << "Number of degrees of freedom: "
              << n_u +  n_T
              << " (" << n_u << '+'<< n_T <<')'
              << std::endl
              << std::endl;

    // The next step is to create the sparsity pattern for the Stokes and
    // concentration system matrices as well as the preconditioner matrix from
    {
        poisson_matrix.clear ();

        DynamicSparsityPattern dsp ( potential_dof_handler.n_dofs(),
                                     potential_dof_handler.n_dofs());

        DoFTools::make_sparsity_pattern (potential_dof_handler, dsp);
        poisson_constraints.condense(dsp);

        poisson_sparsity_pattern.copy_from(dsp);
        poisson_matrix.reinit (poisson_sparsity_pattern);
    }


    {
        concentration_mass_matrix.clear ();
        concentration_matrix.clear ();

        CompressedSparsityPattern csp(concentration_dof_handler.n_dofs());
        DoFTools::make_flux_sparsity_pattern (concentration_dof_handler, csp);
        DG_sparsity_pattern.copy_from(csp);

        DG_concentration_matrix.reinit (DG_sparsity_pattern);
        DG_concentration_mass_matrix.reinit (DG_sparsity_pattern);
        DG_concentration_advec_matrix.reinit (DG_sparsity_pattern);
    }

    // Lastly, we set the vectors for the Stokes solutions $\mathbf u^{n-1}$
    // and $\mathbf u^{n-2}$, as well as for the concentrations $T^{n}$,
    // $T^{n-1}$ and $T^{n-2}$ (required for time stepping) and all the system
    // right hand sides to their correct sizes and block structure:
    potential_solution.reinit (potential_dof_handler.n_dofs());
    old_potential_solution.reinit (potential_dof_handler.n_dofs());
    poisson_rhs.reinit (potential_dof_handler.n_dofs());

    DG_concentration_solution.reinit (concentration_dof_handler.n_dofs());
    DG_old_concentration_solution.reinit (concentration_dof_handler.n_dofs());
    DG_old_old_concentration_solution.reinit (concentration_dof_handler.n_dofs());
    DG_old_old_old_concentration_solution.reinit (concentration_dof_handler.n_dofs());

    DG_concentration_rhs.reinit (concentration_dof_handler.n_dofs());

    concentration_solution=DG_concentration_solution;
    old_concentration_solution=DG_old_concentration_solution;
    old_old_concentration_solution=DG_old_old_concentration_solution;
    old_old_old_concentration_solution=DG_old_old_old_concentration_solution;
    concentration_rhs=DG_concentration_rhs;
}


// @sect4{DriftDiffusionProblem::setup_material_id}
//
template <int dim>
void DriftDiffusionProblem<dim>::setup_material_id ()
{

    typename DoFHandler<dim>::active_cell_iterator
    cell = potential_dof_handler.begin_active(),
    endc = potential_dof_handler.end();
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
// @sect4{DriftDiffusionProblem::apply_bound_preserving_limiter}
//
template <int dim>
void DriftDiffusionProblem<dim>::apply_bound_preserving_limiter ()
{

    typename DoFHandler<dim>::active_cell_iterator
    cell = concentration_dof_handler.begin_active(),
    endc = concentration_dof_handler.end();

    std::cout << "Stat to apply Bound Preserving Limiter "
              << std::endl;
    const QGauss<dim-1> quadrature_formula_x(concentration_degree+1);
    const QGaussLobatto<dim-1> quadrature_formula_y(concentration_degree+1);

    const QGauss<dim> quadrature_formula(concentration_degree+1);
    const QAnisotropic<dim> quadrature_formula_xy(quadrature_formula_x,quadrature_formula_y);
    const QAnisotropic<dim> quadrature_formula_yx(quadrature_formula_y,quadrature_formula_x);

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_q_points_xy = quadrature_formula_xy.size();
    const unsigned int n_q_points_yx = quadrature_formula_yx.size();

    FEValues<dim>  concentration_fe_values (concentration_fe, quadrature_formula,
                                          update_values    |
                                          update_gradients |
                                          update_hessians  |
                                          update_quadrature_points  |
                                          update_JxW_values);
    std::vector<double>   concentration_values(n_q_points);

    FEValues<dim>  concentration_fe_values_xy (concentration_fe, quadrature_formula_xy,
            update_values    |
            update_gradients |
            update_hessians  |
            update_quadrature_points  |
            update_JxW_values);
    std::vector<double>   concentration_values_xy (n_q_points_xy);

    FEValues<dim>  concentration_fe_values_yx (concentration_fe, quadrature_formula_yx,
            update_values    |
            update_gradients |
            update_hessians  |
            update_quadrature_points  |
            update_JxW_values);
    std::vector<double>   concentration_values_yx (n_q_points_yx);

    const unsigned int dofs_per_cell   = concentration_fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    global_concentrational_integrals=0;
    local_concentrational_integrals=0;
    for (; cell!=endc; ++cell)
    {
        cell->get_dof_indices (local_dof_indices);
        double min_concentration = DG_concentration_solution(local_dof_indices[0]),
               max_concentration = DG_concentration_solution(local_dof_indices[0]);
        for (unsigned int i=1; i<local_dof_indices.size(); ++i)
        {
            min_concentration = std::min<double> (min_concentration,
                                                DG_concentration_solution(local_dof_indices[i]));
            max_concentration = std::max<double> (max_concentration,
                                                DG_concentration_solution(local_dof_indices[i]));
        }
        //Find the trouble cell
        if(
            (max_concentration>=(EquationData::Max_T+EquationData::err_tol))
            ||
            (min_concentration<=(EquationData::Min_T-EquationData::err_tol))
        )
        {

            //Find the average of solution on this trouble cell
            double T_cell_average=0;
            double T_cell_area=0;

            concentration_fe_values_xy.reinit (cell);
            concentration_fe_values_xy.get_function_values (DG_concentration_solution,
                    concentration_values_xy);
            for (unsigned int q=0; q<n_q_points_xy; ++q)
            {
                T_cell_average+= concentration_values_xy[q]
                                 *concentration_fe_values_xy.JxW(q);
                T_cell_area+= 1.0
                              *concentration_fe_values_xy.JxW(q);
                min_concentration = std::min<double> (min_concentration,
                                                    concentration_values_xy[q]);
                max_concentration = std::max<double> (max_concentration,
                                                    concentration_values_xy[q]);
            }

            // Need to divide the area of the cell
            T_cell_average/=T_cell_area;

            for (unsigned int q=0; q<n_q_points_yx; ++q)
            {
                min_concentration = std::min<double> (min_concentration,
                                                    concentration_values_yx[q]);
                max_concentration = std::max<double> (max_concentration,
                                                    concentration_values_yx[q]);
            }

            //Define theta
            double theta_T=std::min<double>(1,abs((EquationData::Max_T-T_cell_average)/(max_concentration-T_cell_average)));
            theta_T=std::min<double>(theta_T,abs((EquationData::Min_T-T_cell_average)/(min_concentration-T_cell_average)));
            //Modify the numerical solution at quadrature points
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                double t_tmp=DG_concentration_solution(local_dof_indices[i]);
                DG_concentration_solution(local_dof_indices[i])=theta_T*(t_tmp-T_cell_average)+T_cell_average;
            }
            /*
            //check the cell average again
            T_cell_average=0;

            for (unsigned int q=0; q<n_q_points_xy; ++q)
            {
            T_cell_average+= concentration_values_xy[q]
             *concentration_fe_values_xy.JxW(q);
             }
            // Need to divide the area of the cell
            T_cell_average/=T_cell_area;
            */

        }
    }
        concentration_solution=DG_concentration_solution;
}







// @sect4{DriftDiffusionProblem::assemble_poisson_system}
//
template <int dim>
void DriftDiffusionProblem<dim>::assemble_poisson_system ()
{
    std::cout << "   Assembling..." << std::flush;

    if (rebuild_poisson_matrix == true)
        poisson_matrix=0;

    poisson_rhs=0;

    const QGauss<dim> quadrature_formula (potential_degree+2);
    FEValues<dim>     potential_fe_values (potential_fe, quadrature_formula,
                                        update_values    |
                                        update_quadrature_points  |
                                        update_JxW_values |
                                        (rebuild_poisson_matrix == true
                                         ?
                                         update_gradients
                                         :
                                         UpdateFlags(0)));

    FEValues<dim> concentration_fe_values (concentration_fe, quadrature_formula,
                                         update_values);
    

    // Next we need a vector that will contain the values of the concentration
    MatrixCreator::create_laplace_matrix(potential_dof_handler,
                                         QGauss<dim>(potential_fe.degree+1),
                                         poisson_matrix);// The last two declarations are used to extract the individual blocks


}




// @sect4{DriftDiffusionProblem::assemble_concentration_matrix}
//
template <int dim>
void DriftDiffusionProblem<dim>::assemble_concentration_matrix ()
{

    DG_concentration_mass_matrix = 0;

    MeshWorker::IntegrationInfoBox<dim> info_box;

    const unsigned int n_gauss_points = concentration_dof_handler.get_fe().degree+1;
    info_box.initialize_gauss_quadrature(n_gauss_points,
                                         n_gauss_points,
                                         n_gauss_points);

    info_box.initialize_update_flags();
    UpdateFlags update_flags = update_quadrature_points |
                               update_values            |
                               update_gradients;
    info_box.add_update_flags(update_flags, true, true, true, true);

    info_box.initialize(concentration_fe, mapping);

    Vector<double> rhs_tmp (concentration_dof_handler.n_dofs());

    MeshWorker::DoFInfo<dim> dof_info(concentration_dof_handler);


    DG_concentration_mass_matrix.reinit (DG_sparsity_pattern);
    MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double>> assembler;
    assembler.initialize(DG_concentration_mass_matrix,rhs_tmp);

    auto integrate_cell_term_mass_bind=std::bind(&DriftDiffusionProblem<dim>::integrate_cell_term_mass,this, std::placeholders::_1, std::placeholders::_2, this->potential_solution);
    MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
    (concentration_dof_handler.begin_active(), concentration_dof_handler.end(),
     dof_info, info_box,
     integrate_cell_term_mass_bind,
     NULL,
     NULL,
     assembler);

    //DG_concentration_rhs=rhs_tmp;//if there is a source term;//FIXME
    //copy the DG mass matrix to Trillinor mass matrix
    concentration_mass_matrix.reinit(DG_concentration_mass_matrix);

    // obtain advection matrix
    DG_concentration_advec_matrix.reinit (DG_sparsity_pattern);
    rhs_tmp.reinit(concentration_dof_handler.n_dofs());
    MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> > assembler1;
    assembler1.initialize(DG_concentration_advec_matrix, rhs_tmp);

    //bind definitions
    auto integrate_cell_term_advection_bind = std::bind(&DriftDiffusionProblem<dim>::integrate_cell_term_advection,
            this, std::placeholders::_1, std::placeholders::_2, this->potential_solution);
    auto integrate_boundary_term_advection_bind = std::bind(&DriftDiffusionProblem<dim>::integrate_boundary_term_advection,
            this, std::placeholders::_1, std::placeholders::_2, this->potential_solution);
    auto integrate_face_term_advection_bind = std::bind(&DriftDiffusionProblem<dim>::integrate_face_term_advection,
            this, std::placeholders::_1, std::placeholders::_2,std::placeholders::_3, std::placeholders::_4, this->potential_solution);

    MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
    (concentration_dof_handler.begin_active(), concentration_dof_handler.end(),
     dof_info, info_box,
     integrate_cell_term_advection_bind,
     integrate_boundary_term_advection_bind,
     integrate_face_term_advection_bind,
     assembler1);

    concentration_mass_matrix.reinit(DG_concentration_mass_matrix);
}



// @sect4{DriftDiffusionProblem::solve}
//
template <int dim>
void DriftDiffusionProblem<dim>::solve ()
{
    std::cout << "   Solving..." << std::endl;

    {

        SolverControl solver_control (poisson_matrix.m(),
                                      1e-6*poisson_rhs.l2_norm());
        SolverCG<> cg(solver_control);

        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(poisson_matrix, 1.0);
        
        poisson_rhs*=-1.0;
        
        cg.solve(poisson_matrix, potential_solution, poisson_rhs, preconditioner);

        poisson_constraints.distribute (potential_solution);

        std::cout << "   "
                  << solver_control.last_step()
                  << " GMRES iterations for Stokes subsystem."
                  << std::endl;
    }
    // Once we know the Stokes solution, we can determine the new time step

    old_time_step = time_step;
    const double maximal_potential = get_maximal_potential();

#if 1
    if (maximal_potential >= 0.0001)
        //time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
        //          concentration_degree *
        time_step = 0.05*GridTools::minimal_cell_diameter(triangulation) /
                    maximal_potential;
    else
        //time_step = 1./(1.7*dim*std::sqrt(1.*dim)) /
        //         concentration_degree *
        time_step = 0.05*GridTools::minimal_cell_diameter(triangulation) /
                    .0001;
#else
    time_step = 2;
#endif
    std::cout << "   " << "Time step: " << time_step
              << std::endl;


    // Next we set up the concentration system and the right hand side using the
    {
        if (timestep_number<=1)
        {   // SSP RK Stage 1
            Vector<double> sol_tmp (concentration_dof_handler.n_dofs());
            DG_concentration_advec_matrix.vmult(DG_concentration_rhs,DG_old_concentration_solution);
            DG_concentration_rhs*=-time_step;
            DG_concentration_mass_matrix.vmult(sol_tmp,DG_old_concentration_solution);
            DG_concentration_rhs+=sol_tmp;
            SolverControl solver_control (DG_concentration_mass_matrix.m(),
                                          1e-8*DG_concentration_rhs.l2_norm());
            SolverCG<> cg (solver_control);

            PreconditionBlockSSOR<SparseMatrix<double> > preconditioner;
            preconditioner.initialize(DG_concentration_mass_matrix, concentration_fe.dofs_per_cell);

            cg.solve (DG_concentration_mass_matrix, DG_concentration_solution,
                      DG_concentration_rhs, preconditioner);

            std::cout << "   "
                      << solver_control.last_step()
                      << " CG iterations 1 for concentration."
                      << std::endl;
            apply_bound_preserving_limiter();
            // SSP RK Stage 2
            DG_concentration_advec_matrix.vmult(DG_concentration_rhs,DG_concentration_solution);
            DG_concentration_rhs*=-time_step*0.5;
            DG_old_concentration_solution+=DG_concentration_solution;
            DG_old_concentration_solution*=0.5;
            DG_concentration_mass_matrix.vmult(sol_tmp,DG_old_concentration_solution);
            DG_concentration_rhs+=sol_tmp;

            preconditioner.initialize(DG_concentration_mass_matrix, concentration_fe.dofs_per_cell);
            cg.solve (DG_concentration_mass_matrix, DG_concentration_solution,
                      DG_concentration_rhs, preconditioner);

            std::cout << "   "
                      << solver_control.last_step()
                      << " CG iterations 2 for concentration."
                      << std::endl;
        }
        else
        {   // SSP 2nd multistep method
            DG_concentration_advec_matrix.vmult(DG_concentration_rhs,DG_old_concentration_solution);
            DG_concentration_rhs*=-time_step*1.5;

            Vector<double> rhs_tmp (concentration_dof_handler.n_dofs());

            DG_concentration_mass_matrix.vmult(rhs_tmp,DG_old_old_old_concentration_solution);
            rhs_tmp*=0.25;
            DG_concentration_rhs+=rhs_tmp;
            DG_concentration_mass_matrix.vmult(rhs_tmp,DG_old_concentration_solution);
            rhs_tmp*=0.75;
            DG_concentration_rhs+=rhs_tmp;

            SolverControl solver_control (DG_concentration_mass_matrix.m(),
                                          1e-8*DG_concentration_rhs.l2_norm());
            SolverCG<> cg (solver_control);
            PreconditionBlockSSOR<SparseMatrix<double> > preconditioner;
            preconditioner.initialize(DG_concentration_mass_matrix, concentration_fe.dofs_per_cell);
            cg.solve (DG_concentration_mass_matrix, DG_concentration_solution,
                      DG_concentration_rhs, preconditioner);

            std::cout << "   "
                      << solver_control.last_step()
                      << " CG iterations SSP stage 2 for concentration."
                      << std::endl;
        }

        concentration_solution=DG_concentration_solution;

    }
}



// @sect4{DriftDiffusionProblem::output_results}
template <int dim>
void DriftDiffusionProblem<dim>::output_results ()  const
{
    if (timestep_number % 20 != 0)
        return;

    std::vector<std::string> poisson_names (dim, "potential");
    poisson_names.push_back ("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    poisson_component_interpretation
    (dim+1, DataComponentInterpretation::component_is_scalar);
    for (unsigned int i=0; i<dim; ++i)
        poisson_component_interpretation[i]
            = DataComponentInterpretation::component_is_part_of_vector;

    DataOut<dim> data_out;
    data_out.add_data_vector (potential_dof_handler, potential_solution,
                              poisson_names, poisson_component_interpretation);
    data_out.add_data_vector (concentration_dof_handler, concentration_solution,
                              "C");
    data_out.add_data_vector (concentration_dof_handler, DG_concentration_solution,
                              "DG_C");
    data_out.build_patches (std::min(potential_degree, concentration_degree));

    std::ostringstream filename;
    filename << "solution-" << Utilities::int_to_string(timestep_number, 4) << ".vtk";
    //filename << "solution-" << Utilities::int_to_string(timestep_number, 4) << ".gnuplot";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
    //data_out.write_gnuplot (output);
}



// @sect4{DriftDiffusionProblem::refine_mesh}
//
template <int dim>
void DriftDiffusionProblem<dim>::refine_mesh (const unsigned int max_grid_level)
{
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (concentration_dof_handler,
                                        QGauss<dim-1>(concentration_degree+1),
                                        typename FunctionMap<dim>::type(),
                                        concentration_solution,
                                        estimated_error_per_cell);

#if 1
    GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
          estimated_error_per_cell,
        .99, 0.01);
#else
    GridRefinement::refine(triangulation,
                           estimated_error_per_cell,
                           .1);

#endif
    if (triangulation.n_levels() > max_grid_level)
        for (typename Triangulation<dim>::active_cell_iterator
                cell = triangulation.begin_active(max_grid_level);
                cell != triangulation.end(); ++cell)
            cell->clear_refine_flag ();

    //
    // Consequently, we initialize two SolutionTransfer objects for the Stokes
    // and concentration DoFHandler objects, by attaching them to the old dof
    // handlers. With this at place, we can prepare the triangulation and the
    // data vectors for refinement (in this order).
    std::vector<TrilinosWrappers::Vector> x_concentration (3);
    x_concentration[0] = concentration_solution;
    x_concentration[1] = old_concentration_solution;
    x_concentration[2] = old_old_concentration_solution;
    TrilinosWrappers::BlockVector x_poisson = potential_solution;

    SolutionTransfer<dim,TrilinosWrappers::Vector>
    concentration_trans(concentration_dof_handler);
    SolutionTransfer<dim,TrilinosWrappers::BlockVector>
    poisson_trans(potential_dof_handler);

    triangulation.prepare_coarsening_and_refinement();
    concentration_trans.prepare_for_coarsening_and_refinement(x_concentration);
    poisson_trans.prepare_for_coarsening_and_refinement(x_poisson);

    // Now everything is ready, so do the refinement and recreate the dof
    // structure on the new grid, and initialize the matrix structures and the
    triangulation.execute_coarsening_and_refinement ();
    setup_dofs ();

    std::vector<TrilinosWrappers::Vector> tmp (3);
    tmp[0].reinit (concentration_solution);
    tmp[1].reinit (concentration_solution);
    tmp[2].reinit (concentration_solution);
    concentration_trans.interpolate(x_concentration, tmp);

    concentration_solution = tmp[0];
    old_concentration_solution = tmp[1];
    old_old_concentration_solution = tmp[2];

    DG_concentration_solution = tmp[0];
    DG_old_concentration_solution = tmp[1];
    DG_old_old_concentration_solution = tmp[2];

    poisson_trans.interpolate (x_poisson, potential_solution);

    rebuild_poisson_matrix         = true;
    rebuild_concentration_matrices  = true;
    rebuild_poisson_preconditioner = true;
}



// @sect4{DriftDiffusionProblem::run}
//
template <int dim>
void DriftDiffusionProblem<dim>::run ()
{
    const unsigned int initial_refinement = (dim == 2 ? 4 : 2);
    const unsigned int n_pre_refinement_steps = (dim == 2 ? 5 : 3);


    GridGenerator::hyper_cube (triangulation,
                               0.,
                               1.);
    global_Omega_diameter = GridTools::diameter (triangulation);
//    triangulation.refine_global (initial_refinement);
    triangulation.refine_global (8); //if want to use global uniform mesh

    setup_dofs();
    setup_material_id();
    unsigned int pre_refinement_step = 0;

start_time_iteration:

    const QIterated<dim> quadrature_formula (QTrapez<1>(),
            concentration_degree);

    FEValues<dim> fe_values (concentration_fe, quadrature_formula,
                             update_values | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell   = concentration_fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    std::vector<double>  local_vector(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = concentration_dof_handler.begin_active(),
    endc = concentration_dof_handler.end();

    EquationData::CompositionInitialValues<dim> concentration_intial;

    for (; cell!=endc; ++cell)
    {
        fe_values.reinit(cell);
        cell->get_dof_indices (local_dof_indices);
        for (unsigned int q=0; q<n_q_points; ++q)
        {
            local_vector[q]=cell->material_id();
            //local_vector[q]=concentration_intial.value(fe_values.quadrature_point(q),0);
        }
        old_concentration_solution.set(local_dof_indices,local_vector);
    }
    DG_old_concentration_solution=old_concentration_solution;
    DG_old_old_concentration_solution=old_concentration_solution;
    DG_concentration_solution=old_concentration_solution;

    timestep_number           = 0;
    time_step = old_time_step = 0;

    time =0;
    output_results ();
#if 0
    do
    {
        std::cout << "Timestep " << timestep_number
                  << ":  t=" << time
                  << std::endl;

        // The first steps in the time loop are all obvious &ndash; we
        assemble_poisson_system ();
        build_poisson_preconditioner ();
        assemble_concentration_matrix ();

        solve ();
        apply_bound_preserving_limiter();
        double min_concentration = DG_concentration_solution(0),
               max_concentration = DG_concentration_solution(0);
        for (unsigned int i=0; i<DG_concentration_solution.size(); ++i)
        {
            min_concentration = std::min<double> (min_concentration,
                                                DG_concentration_solution(i));
            max_concentration = std::max<double> (max_concentration,
                                                DG_concentration_solution(i));
        }

        std::cout << "   Composition range1: "
                  << min_concentration << ' ' << max_concentration
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
                  << global_concentrational_integrals
                  << std::endl;

        time += time_step;
        ++timestep_number;
        output_results ();

        old_potential_solution             = potential_solution;
        old_old_old_concentration_solution = old_old_concentration_solution;
        old_old_concentration_solution    = old_concentration_solution;
        old_concentration_solution     = concentration_solution;

        DG_old_old_old_concentration_solution = DG_old_old_concentration_solution;
        DG_old_old_concentration_solution    = DG_old_concentration_solution;
        DG_old_concentration_solution     = DG_concentration_solution;

        rebuild_poisson_matrix         = true;
        rebuild_concentration_matrices  = true;
        rebuild_poisson_preconditioner = true;

        min_concentration = DG_concentration_solution(0);
        max_concentration = DG_concentration_solution(0);
        for (unsigned int i=0; i<DG_concentration_solution.size(); ++i)
        {
            min_concentration = std::min<double> (min_concentration,
                                                DG_concentration_solution(i));
            max_concentration = std::max<double> (max_concentration,
                                                DG_concentration_solution(i));
        }

        std::cout << "   Composition range2: "
                  << min_concentration << ' ' << max_concentration
                  << std::endl;
        std::cout << "   Composition mass integral: "
                  << global_concentrational_integrals
                  << std::endl;
#ifdef OUTPUT_FILE
        output_file_m << global_concentrational_integrals << std::endl;
        output_file_c << min_concentration << ' ' << max_concentration
                      << std::endl;
#endif
    }
    // Do all the above until we arrive at time 100.
    while (timestep_number <= 5001);
#endif
}
}



// @sect3{The <code>main</code> function}
//
int main (int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace fem_dg;

        deallog.depth_console (0);

        Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
                numbers::invalid_unsigned_int);

        DriftDiffusionProblem<2> drift_problem;
        drift_problem.run ();
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
