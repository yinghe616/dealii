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
  const double Max_C = 1;
  const double Min_C = 0;
  const double elementary_charge = 1e0;
  const double mobility_neg        = 16;
  const double mobility_pos        = 16;
  const double diffusivity_neg     = 0.4;
  const double diffusivity_pos     = 0.4;
  const double permitivity         = 30;
  const double density             = 1e4;
  const double err_tol = 1e-10;
//define mobility
  template <int dim>
    class MobilityValues : public Function<dim>
  {
    public:
      MobilityValues () : Function<dim>(1) {}
      virtual double value (const Point<dim>   &p,
          const unsigned int  component = 0) const;
      virtual void vector_value (const Point<dim> &p,
          Vector<double>   &value) const;
  };
  template <int dim>
    double
    MobilityValues<dim>::value (const Point<dim> &p,
        const unsigned int) const
    {
      return (((p(0)<=1)&&(p(0)>=-1)) ? 16:60);
    }

  template <int dim>
    void
    MobilityValues<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = MobilityValues<dim>::value (p, c);
    }

  //define permitivity
  template <int dim>
    class PermitivityValues : public Function<dim>
  {
    public:
      PermitivityValues () : Function<dim>(1) {}
      virtual double value (const Point<dim>   &p,
          const unsigned int  component = 0) const;
      virtual void vector_value (const Point<dim> &p,
          Vector<double>   &value) const;
  };
  template <int dim>
    double
    PermitivityValues<dim>::value (const Point<dim> &p,
        const unsigned int) const
    {
      return ((p(0)<=1)&&(p(0)>=-1) ? 30:80);
    }

  template <int dim>
    void
    PermitivityValues<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = PermitivityValues<dim>::value (p, c);
    }

  //define diffusivity
  template <int dim>
    class DiffusivityValues : public Function<dim>
  {
    public:
      DiffusivityValues () : Function<dim>(1) {}
      virtual double value (const Point<dim>   &p,
          const unsigned int  component = 0) const;
      virtual void vector_value (const Point<dim> &p,
          Vector<double>   &value) const;
  };
  template <int dim>
    double
    DiffusivityValues<dim>::value (const Point<dim> &p,
        const unsigned int) const
    {
      return ((p(0)<=1)&&(p(0)>=-1) ? 0.4:1.5);
    }

  template <int dim>
    void
    DiffusivityValues<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = DiffusivityValues<dim>::value (p, c);
  
    }
  //define initddial values

  template <int dim>
    class ConcentrationNegativeInitialValues : public Function<dim>
  {
    public:
      ConcentrationNegativeInitialValues () : Function<dim>(1) {}

      virtual double value (const Point<dim>   &p,
          const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
          Vector<double>   &value) const;
  };


  template <int dim>
    double
    ConcentrationNegativeInitialValues<dim>::value (const Point<dim> &p,
        const unsigned int) const
    {
      return ((p(1)<=1)&& (p(1)>=-1)&&(p(0)<=1)&&(p(0)>=-1) ? 0*EquationData::density:EquationData::density);
      const double pi=3.1415926;
      //return sin(2.0*pi*p(1))*sin(2.0*pi*p(0));
    }

  template <int dim>
    void
    ConcentrationNegativeInitialValues<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = ConcentrationNegativeInitialValues<dim>::value (p, c);
    }

  template <int dim>
    class ConcentrationPositiveInitialValues : public Function<dim>
  {
    public:
      ConcentrationPositiveInitialValues () : Function<dim>(1) {}

      virtual double value (const Point<dim>   &p,
          const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
          Vector<double>   &value) const;
  };


  template <int dim>
    double
    ConcentrationPositiveInitialValues<dim>::value (const Point<dim> &p,
        const unsigned int) const
    { double return_value;
      if (p(0) <=-1 || p(0)>=1)
      return_value = EquationData::density;
      else if (p(0) <= .25 && p(0)>=-.25)
      return_value = 1.2*EquationData::density;

      return return_value;
    }


  template <int dim>
    void
    ConcentrationPositiveInitialValues<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = ConcentrationPositiveInitialValues<dim>::value (p, c);
    }


  // define concentration negative right hand side function class
  template <int dim>
    class ConcentrationNegativeRightHandSide : public Function<dim>
  {
    public:
      ConcentrationNegativeRightHandSide () : Function<dim>(1) {}

      virtual double value (const Point<dim>   &p,
          const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
          Vector<double>   &value) const;
  };


  template <int dim>
    double
    ConcentrationNegativeRightHandSide<dim>::value (const Point<dim>  &p,
        const unsigned int component) const
    {
      Assert (component == 0,
          ExcMessage ("Invalid operation for a scalar function."));

      Assert ((dim==2) || (dim==3), ExcNotImplemented());

      return 0;
    }


  template <int dim>
    void
    ConcentrationNegativeRightHandSide<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = ConcentrationNegativeRightHandSide<dim>::value (p, c);
    }

  // define concentration positive right hand side function class
  template <int dim>
    class ConcentrationPositiveRightHandSide : public Function<dim>
  {
    public:
      ConcentrationPositiveRightHandSide () : Function<dim>(1) {}

      virtual double value (const Point<dim>   &p,
          const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
          Vector<double>   &value) const;
  };


  template <int dim>
    double
    ConcentrationPositiveRightHandSide<dim>::value (const Point<dim>  &p,
        const unsigned int component) const
    {
      Assert (component == 0,
          ExcMessage ("Invalid operation for a scalar function."));

      Assert ((dim==2) || (dim==3), ExcNotImplemented());

      return 0;
    }


  template <int dim>
    void
    ConcentrationPositiveRightHandSide<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = ConcentrationPositiveRightHandSide<dim>::value (p, c);
    }
  
  // define potential right hand side function class
  template <int dim>
    class PotentialRightHandSide : public Function<dim>
  {
    public:
      PotentialRightHandSide () : Function<dim>(1) {}

      virtual double value (const Point<dim>   &p,
          const unsigned int  component = 0) const;

      virtual void vector_value (const Point<dim> &p,
          Vector<double>   &value) const;
  };


  template <int dim>
    double
    PotentialRightHandSide<dim>::value (const Point<dim>  &p,
        const unsigned int component) const
    {
      Assert (component == 0,
          ExcMessage ("Invalid operation for a scalar function."));

      Assert ((dim==2) || (dim==3), ExcNotImplemented());
      double return_value = 0;
      if (dim ==2)
       //  return_value =0.1*(p(0)*p(0)-1)*(p(1)*p(1)-1)*std::exp(p(0)*p(0)+p(1)*p(1));
    //  return return_value;
      return ((p(0)<=.25)&&(p(0)>=-.25) ? -1.2*EquationData::density:0);
    }


  template <int dim>
    void
    PotentialRightHandSide<dim>::vector_value (const Point<dim> &p,
        Vector<double>   &values) const
    {
      for (unsigned int c=0; c<this->n_components; ++c)
        values(c) = PotentialRightHandSide<dim>::value (p, c);
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
    void setup_boundary_ids ();
    void apply_bound_preserving_limiter ();
    void assemble_poisson_system ();
    void assemble_concentration_matrix ();
    double get_maximal_potential () const;
    std::pair<double,double> get_extrapolated_concentration_range () const;
    void solve_poisson ();
    void solve_concentration ();
    void output_results () const;
    void refine_mesh (const unsigned int max_grid_level);

    Triangulation<dim>                  triangulation;
    const MappingQ1<dim>                mapping;
    double                              global_Omega_diameter;

    const unsigned int                  potential_degree;
    FE_Q<dim>                           potential_fe;
    DoFHandler<dim>                     potential_dof_handler;
    ConstraintMatrix                    poisson_constraints;
    SparsityPattern                     poisson_sparsity_pattern;

    SparseMatrix<double>                poisson_matrix;
    SparseMatrix<double>                poisson_mass_matrix;

    Vector<double>                      poisson_rhs;
    Vector<double>                      potential_solution;
    Vector<double>                      old_potential_solution;


    const unsigned int                  concentration_degree;
    FE_Q<dim>                           concentration_fe;
    DoFHandler<dim>                     concentration_dof_handler;
    ConstraintMatrix                    concentration_constraints;
    SparsityPattern                     concentration_sparsity_pattern;

    SparseMatrix<double>                concentration_mass_matrix;
    SparseMatrix<double>                concentration_laplace_matrix;
    SparseMatrix<double>                concentration_matrix_neg;
    SparseMatrix<double>                concentration_matrix_pos;


    Vector<double>                      concentration_solution_neg;
    Vector<double>                      old_concentration_solution_neg;
    Vector<double>                      old_old_concentration_solution_neg;
    Vector<double>                      old_old_old_concentration_solution_neg;
    Vector<double>                      concentration_rhs_neg;

    Vector<double>                      concentration_solution_pos;
    Vector<double>                      old_concentration_solution_pos;
    Vector<double>                      old_old_concentration_solution_pos;
    Vector<double>                      old_old_old_concentration_solution_pos;
    Vector<double>                      concentration_rhs_pos;
    double                              time;
    double                              time_step;
    double                              old_time_step;
    unsigned int                        timestep_number;

    double                              global_concentrational_integrals;
    double                              local_concentrational_integrals;

    bool                                rebuild_poisson_matrix;
    bool                                rebuild_concentration_matrices;

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
                                   Vector<double> &coef);
    void integrate_cell_term_advection (DoFInfo &dinfo,
                                        CellInfo &info,
                                        Vector<double> &coef);
    void integrate_cell_term_source (DoFInfo &dinfo,
                                     CellInfo &info,
                                     Vector<double> &coef);
    void integrate_boundary_term_advection (DoFInfo &dinfo,
                                            CellInfo &info,
                                            Vector<double> &coef);
    void integrate_face_term_advection (DoFInfo &dinfo1,
                                        DoFInfo &dinfo2,
                                        CellInfo &info1,
                                        CellInfo &info2,
                                        Vector<double> &coef);
};

// @sect4{The local integrators}

// These are the functions given to the MeshWorker::integration_loop()
// called just above. They compute the local contributions to the system
// matrix and right hand side on cells and faces.
template <int dim>
void DriftDiffusionProblem<dim>::integrate_cell_term_advection (DoFInfo &dinfo,
        CellInfo &info,
        Vector<double> &coef)
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
    std::vector<double> potential_values(n_q_points);

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
        Vector<double> &coef)
{
}


template <int dim>
void DriftDiffusionProblem<dim>::integrate_cell_term_mass (DoFInfo &dinfo,
        CellInfo &info,
        Vector<double> &coef)
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
    EquationData::ConcentrationNegativeRightHandSide<dim>  concentration_neg_right_hand_side;
    concentration_neg_right_hand_side.value_list (fe_v.get_quadrature_points(), g);

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
        Vector<double> &coef)
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

    std::vector<Tensor<1,dim> > potential_gradients_values(n_q_points);
    potential_fe_values.get_function_gradients (coef,
            potential_gradients_values);

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        const double beta_n = potential_gradients_values[point]* normals[point];
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
        Vector<double> &coef)
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

    std::vector<Tensor<1,dim> > potential_gradients_values(n_q_points);

    potential_fe_values.get_function_gradients (coef,
            potential_gradients_values);

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        const double beta_n=potential_gradients_values[point] * normals[point];
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
    rebuild_concentration_matrices (true)
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
            fe_values.get_function_values (old_concentration_solution_neg,
                                           old_concentration_values);
            fe_values.get_function_values (old_old_concentration_solution_neg,
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
            fe_values.get_function_values (old_concentration_solution_neg,
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
        poisson_mass_matrix.clear ();

        DynamicSparsityPattern dsp ( potential_dof_handler.n_dofs(),
                                     potential_dof_handler.n_dofs());

        DoFTools::make_sparsity_pattern (potential_dof_handler, dsp);
        poisson_constraints.condense(dsp);

        poisson_sparsity_pattern.copy_from(dsp);
        poisson_matrix.reinit (poisson_sparsity_pattern);
        poisson_mass_matrix.reinit (poisson_sparsity_pattern);
    }


    {
        concentration_mass_matrix.clear ();
        concentration_laplace_matrix.clear ();
        concentration_matrix_neg.clear ();
        concentration_matrix_pos.clear ();

        DynamicSparsityPattern dsp2 ( concentration_dof_handler.n_dofs(),
                                      concentration_dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (concentration_dof_handler, dsp2);
        concentration_constraints.condense(dsp2);

        concentration_sparsity_pattern.copy_from(dsp2);
        concentration_mass_matrix.reinit (concentration_sparsity_pattern);
        concentration_laplace_matrix.reinit (concentration_sparsity_pattern);
        concentration_matrix_neg.reinit (concentration_sparsity_pattern);
        concentration_matrix_pos.reinit (concentration_sparsity_pattern);
    }

    // Lastly, we set the vectors for the Stokes solutions $\mathbf u^{n-1}$
    // and $\mathbf u^{n-2}$, as well as for the concentrations $T^{n}$,
    // $T^{n-1}$ and $T^{n-2}$ (required for time stepping) and all the system
    // right hand sides to their correct sizes and block structure:
    potential_solution.reinit (potential_dof_handler.n_dofs());
    old_potential_solution.reinit (potential_dof_handler.n_dofs());
    poisson_rhs.reinit (potential_dof_handler.n_dofs());

    concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());
    old_concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());
    old_old_concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());
    old_old_old_concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());

    concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());
    old_concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());
    old_old_concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());
    old_old_old_concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());

    concentration_rhs_neg.reinit (concentration_dof_handler.n_dofs());
    concentration_rhs_pos.reinit (concentration_dof_handler.n_dofs());
}


// @sect4{DriftDiffusionProblem::setup_boundary_ids}
  template <int dim>
void DriftDiffusionProblem<dim>::setup_boundary_ids ()
{
  typename DoFHandler<dim>::active_cell_iterator
    cell = potential_dof_handler.begin_active(),
         endc = potential_dof_handler.end();
  for (; cell!=endc; ++cell)
    for (unsigned int face_number=0;
        face_number<GeometryInfo<dim>::faces_per_cell;
        ++face_number)
      if (cell->face(face_number)->at_boundary())
      {  
        {
          if (std::fabs(cell->face(face_number)->center()[0] - (-6)) < 1e-12)
            cell->face(face_number)->set_boundary_id (1);
          if (std::fabs(cell->face(face_number)->center()[0] - (6)) < 1e-12)
            cell->face(face_number)->set_boundary_id (2);
          if (std::fabs(cell->face(face_number)->center()[1] - (6)) < 1e-12)
            cell->face(face_number)->set_boundary_id (3);
          if (std::fabs(cell->face(face_number)->center()[1] - (-6)) < 1e-12)
            cell->face(face_number)->set_boundary_id (3);
        }
      }
}


template <int dim>
void DriftDiffusionProblem<dim>::assemble_poisson_system ()
{

  const QGauss<dim> quadrature_formula (potential_degree+2);
  FEValues<dim>     potential_fe_values (potential_fe, quadrature_formula,
      update_values       |
      update_gradients    |
      update_quadrature_points  |
      update_JxW_values);

  FEValues<dim> concentration_fe_values (concentration_fe, quadrature_formula,
      update_values |
      update_quadrature_points  |
      update_JxW_values );

  const EquationData::PotentialRightHandSide<dim> potential_right_hand_side;
  const EquationData::PermitivityValues<dim>     permitivity_func;

  const unsigned int   dofs_per_cell = potential_fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>            cell_rhs (dofs_per_cell);
  std::vector<double>       concentration_values_neg(n_q_points);
  std::vector<double>       concentration_values_pos(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = potential_dof_handler.begin_active(),
         endc = potential_dof_handler.end();

  poisson_rhs=0;
  if (rebuild_poisson_matrix)
  {
    std::cout << "   Assembling poisson..." << std::endl;
    for (; cell!=endc; ++cell)
    {
      cell->get_dof_indices (local_dof_indices);
      potential_fe_values.reinit (cell);
      cell_matrix = 0;
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
          { cell_matrix(i,j) +=
            permitivity_func.value(potential_fe_values.quadrature_point (q_index))*
              potential_fe_values.shape_grad (i, q_index) *
              potential_fe_values.shape_grad (j, q_index) *
              potential_fe_values.JxW (q_index); 
          }
          poisson_matrix.add (local_dof_indices[i],
              local_dof_indices[j],
              cell_matrix(i,j));
        }
    }
    rebuild_poisson_matrix = false;
  }

  cell = potential_dof_handler.begin_active();

  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices);
    potential_fe_values.reinit (cell);
    cell_rhs = 0;
    concentration_fe_values.reinit (cell);
    concentration_fe_values.get_function_values(concentration_solution_neg, concentration_values_neg);
    concentration_fe_values.get_function_values(concentration_solution_pos, concentration_values_pos);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
        cell_rhs(i) += 
          (
           concentration_values_pos[q_index]-concentration_values_neg[q_index]
           +potential_right_hand_side.value(potential_fe_values.quadrature_point (q_index))
          )
          *
          potential_fe_values.shape_value (i, q_index) *potential_fe_values.JxW (q_index);
      }
      poisson_rhs(local_dof_indices[i]) += cell_rhs(i);
    }

  }
  poisson_constraints.condense (poisson_matrix, poisson_rhs); 
  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (potential_dof_handler,
      1,
      ConstantFunction<dim>(.001),
      boundary_values);
  VectorTools::interpolate_boundary_values (potential_dof_handler,
      2,
      ConstantFunction<dim>(-.001),
      boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
      poisson_matrix,
      potential_solution,
      poisson_rhs);

}




// @sect4{DriftDiffusionProblem::assemble_concentration_matrix}
//
template <int dim>
void DriftDiffusionProblem<dim>::assemble_concentration_matrix ()
{
    // Once we know the Stokes solution, we can determine the new time step

    old_time_step = time_step;
    const double maximal_potential = get_maximal_potential();

#if 0
    if (maximal_potential >= 0.0001)
        time_step = 0.1*GridTools::minimal_cell_diameter(triangulation) /
                    maximal_potential;
    else
        time_step = 0.1*GridTools::minimal_cell_diameter(triangulation) /
                    .0001;
#else
    time_step = 0.00001;
#endif


    concentration_rhs_neg=0;
    concentration_rhs_pos=0;

    const QGauss<dim> quadrature_formula (concentration_degree+2);
    FEValues<dim>     poisson_fe_values (potential_fe, quadrature_formula,
                                        update_values       |
                                        update_gradients    |
                                        update_quadrature_points  |
                                        update_JxW_values);

    FEValues<dim> concentration_fe_values (concentration_fe, quadrature_formula,
                                           update_values |
                                        update_gradients    |
                                        update_quadrature_points  |
                                        update_JxW_values );

    const unsigned int   dofs_per_cell = concentration_fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs_neg (dofs_per_cell);
    Vector<double>       cell_rhs_pos (dofs_per_cell);
    std::vector<double>       concentration_neg_values(n_q_points);
    std::vector<double>       concentration_pos_values(n_q_points);
    std::vector< Tensor<1,dim> >       grad_poisson_values(n_q_points);

    const EquationData::MobilityValues<dim>     mobility_func;
    const EquationData::DiffusivityValues<dim>     diffusivity_func;
    
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    typename DoFHandler<dim>::active_cell_iterator
    cell = concentration_dof_handler.begin_active(),
    endc = concentration_dof_handler.end();

    if (rebuild_concentration_matrices)
    {
      rebuild_concentration_matrices = false;
      std::cout << "   Assembling concentration..." << std::endl;
      concentration_matrix_neg=0;
      concentration_matrix_pos=0;
      MatrixCreator::create_mass_matrix(concentration_dof_handler,
          QGauss<dim>(concentration_fe.degree+1),
          concentration_mass_matrix);
      concentration_matrix_neg.copy_from(concentration_mass_matrix);
      concentration_matrix_pos.copy_from(concentration_mass_matrix);
      for (; cell!=endc; ++cell)
      { 
        cell_matrix = 0;
        cell->get_dof_indices (local_dof_indices);
        concentration_fe_values.reinit (cell);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {  for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
            { 
              cell_matrix(i,j) +=
                time_step*
                diffusivity_func.value(concentration_fe_values.quadrature_point (q_index))*
                concentration_fe_values.shape_grad (i, q_index) *
                concentration_fe_values.shape_grad (j, q_index) *
                concentration_fe_values.JxW (q_index); 
            }
            concentration_matrix_neg.add (local_dof_indices[i],
                local_dof_indices[j],
                cell_matrix(i,j));
            concentration_matrix_pos.add (local_dof_indices[i],
                local_dof_indices[j],
                cell_matrix(i,j));
          }
        }
      }
    }

    cell = concentration_dof_handler.begin_active();

    for (; cell!=endc; ++cell)
    {
      cell->get_dof_indices (local_dof_indices);
      concentration_fe_values.reinit (cell);
      poisson_fe_values.reinit (cell);
      poisson_fe_values.get_function_gradients(potential_solution, grad_poisson_values);
      concentration_fe_values.get_function_values(concentration_solution_neg, concentration_neg_values);
      concentration_fe_values.get_function_values(concentration_solution_pos, concentration_pos_values);
      cell_rhs_neg = 0;
      cell_rhs_pos = 0;
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          cell_rhs_neg(i) += 
              time_step*
              (concentration_fe_values.shape_grad (i, q_index) *
              grad_poisson_values[q_index]*
              mobility_func.value(concentration_fe_values.quadrature_point (q_index))*
              concentration_neg_values[q_index]*
              concentration_fe_values.JxW (q_index));
          cell_rhs_pos(i) -=
              time_step*
              (concentration_fe_values.shape_grad (i, q_index) *
              grad_poisson_values[q_index]*
              mobility_func.value(concentration_fe_values.quadrature_point (q_index))*
              concentration_pos_values[q_index]*
              concentration_fe_values.JxW (q_index));
        }
        concentration_rhs_neg(local_dof_indices[i]) += cell_rhs_neg(i);
        concentration_rhs_pos(local_dof_indices[i]) += cell_rhs_pos(i);
      }
    }
    concentration_constraints.condense (concentration_matrix_neg, concentration_rhs_neg); 
    concentration_constraints.condense (concentration_matrix_pos, concentration_rhs_pos); 
  

    std::map<types::global_dof_index,double> boundary_values_neg;
    VectorTools::interpolate_boundary_values (concentration_dof_handler,
                                            1,
                                            ConstantFunction<dim>(EquationData::density),
                                            boundary_values_neg);
    VectorTools::interpolate_boundary_values (concentration_dof_handler,
                                            2,
                                            ConstantFunction<dim>(EquationData::density),
                                            boundary_values_neg);
    VectorTools::interpolate_boundary_values (concentration_dof_handler,
                                            3,
                                            ConstantFunction<dim>(EquationData::density),
                                            boundary_values_neg);
    MatrixTools::apply_boundary_values (boundary_values_neg,
                                      concentration_matrix_neg,
                                      concentration_solution_neg,
                                      concentration_rhs_neg);

    std::map<types::global_dof_index,double> boundary_values_pos;
    VectorTools::interpolate_boundary_values (concentration_dof_handler,
                                            1,
                                            ConstantFunction<dim>(EquationData::density),
                                            boundary_values_pos);
    VectorTools::interpolate_boundary_values (concentration_dof_handler,
                                            2,
                                            ConstantFunction<dim>(EquationData::density),
                                            boundary_values_pos);
    VectorTools::interpolate_boundary_values (concentration_dof_handler,
                                            3,
                                            ConstantFunction<dim>(EquationData::density),
                                            boundary_values_pos);
    MatrixTools::apply_boundary_values (boundary_values_pos,
                                      concentration_matrix_pos,
                                      concentration_solution_pos,
                                      concentration_rhs_pos);
}



// @sect4{DriftDiffusionProblem::solve}
//
template <int dim>
void DriftDiffusionProblem<dim>::solve_poisson ()
{
  std::cout << "   Solving poisson..." << std::endl;
  {
    SolverControl solver_control (poisson_matrix.m(),
        1e-6*poisson_rhs.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(poisson_matrix, 1.0);

    cg.solve(poisson_matrix, potential_solution, poisson_rhs, preconditioner);

    poisson_constraints.distribute (potential_solution);

    std::cout << "   "
      << solver_control.last_step()
      << " CG iterations for poisson equation."
      << std::endl;
  }
}

  
template <int dim>
void DriftDiffusionProblem<dim>::solve_concentration ()
{
  std::cout << "   " << "Time step: " << time_step
    << std::endl;

  std::cout << "   Solving concentration..." << std::endl;
  {
    SolverControl solver_control (concentration_matrix_neg.m(),
        1e-6*concentration_rhs_neg.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(concentration_matrix_neg, 1.0);

    cg.solve(concentration_matrix_neg, concentration_solution_neg, concentration_rhs_neg, preconditioner);

    concentration_constraints.distribute (concentration_solution_neg);

    std::cout << "   "
      << solver_control.last_step()
      << " CG iterations for concentration equation."
      << std::endl;

    SolverControl solver_control_2 (concentration_matrix_pos.m(),
        1e-6*concentration_rhs_pos.l2_norm());
    SolverCG<> cg_2(solver_control_2);

    PreconditionSSOR<> preconditioner_2;
    preconditioner_2.initialize(concentration_matrix_pos, 1.0);

    cg_2.solve(concentration_matrix_pos, concentration_solution_pos, concentration_rhs_pos, preconditioner_2);

    concentration_constraints.distribute (concentration_solution_pos);

    std::cout << "   "
      << solver_control_2.last_step()
      << " CG iterations for concentration equation."
      << std::endl;
  }
}



// @sect4{DriftDiffusionProblem::output_results}
template <int dim>
void DriftDiffusionProblem<dim>::output_results ()  const
{
    if (timestep_number % 1 != 0)
        return;

    DataOut<dim> data_out;
    data_out.add_data_vector (potential_dof_handler, potential_solution,
                              "Potential");
    data_out.add_data_vector (concentration_dof_handler, concentration_solution_neg,
                              "Concentration_negative");
    data_out.add_data_vector (concentration_dof_handler, concentration_solution_pos,
                              "Concentration_positive");
    data_out.build_patches (std::min(potential_degree, concentration_degree));

    std::ostringstream filename;
    filename << "solution-" << Utilities::int_to_string(timestep_number, 4) << ".vtk";
    //filename << "solution-" << Utilities::int_to_string(timestep_number, 4) << ".gnuplot";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
    //data_out.write_gnuplot (output);
}






// @sect4{DriftDiffusionProblem::run}
//
template <int dim>
void DriftDiffusionProblem<dim>::run ()
{
  /*
    const bool colorize = true;
    Triangulation<dim> tria1;
    std::vector< unsigned int> repetitions1(2);
    repetitions1[0]=5;
    repetitions1[1]=12;
    GridGenerator::subdivided_hyper_rectangle(tria1,repetitions1,
                                Point<dim>  (-6,-6),
                                Point<dim>  (-1, 6));
    Triangulation<dim> tria2;
    std::vector< unsigned int> repetitions2(2);
    repetitions2[0]=2;
    repetitions2[1]=2;
    GridGenerator::subdivided_hyper_rectangle(tria2,repetitions2,
                                Point<dim>  (-1,-1),
                                Point<dim>  ( 1, 1));
    Triangulation<dim> tria_merge_1;
    GridGenerator::merge_triangulations (tria1, tria2, tria_merge_1);
    Triangulation<dim> tria3;
    std::vector< unsigned int> repetitions3(2);
    repetitions3[0]=5;
    repetitions3[1]=12;
    GridGenerator::subdivided_hyper_rectangle(tria3,repetitions3,
                                Point<dim>  (1,-6),
                                Point<dim>  (6,6));
    GridGenerator::merge_triangulations (tria_merge_1, tria3, triangulation);
    */
    std::vector< unsigned int> repetitions2(2);
    repetitions2[0]=12;
    repetitions2[1]=12;
    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions2,
                                Point<dim>  (-6,-6),
                                Point<dim>  ( 6, 6));
    global_Omega_diameter = GridTools::diameter (triangulation);
    triangulation.refine_global (3); //if want to use global uniform mesh

    setup_dofs();
    setup_boundary_ids();

start_time_iteration:
    VectorTools::project (concentration_dof_handler,
                          concentration_constraints,
                          QGauss<dim>(concentration_degree+2),
                          EquationData::ConcentrationNegativeInitialValues<dim>(),
                          old_concentration_solution_neg);
    VectorTools::project (concentration_dof_handler,
                          concentration_constraints,
                          QGauss<dim>(concentration_degree+2),
                          EquationData::ConcentrationPositiveInitialValues<dim>(),
                          old_concentration_solution_pos);

    concentration_solution_neg = old_concentration_solution_neg;
    concentration_solution_pos = old_concentration_solution_pos;

    timestep_number           = 0;
    time_step = old_time_step = 0;

    time =0;
    output_results ();
    do
    {
        std::cout << "Timestep " << timestep_number
                  << ":  t=" << time
                  << std::endl;

        assemble_poisson_system ();
        solve_poisson ();

        assemble_concentration_matrix ();
        solve_concentration ();
//        apply_bound_preserving_limiter();


        time += time_step;
        ++timestep_number;
        output_results ();

        old_potential_solution             = potential_solution;
        old_old_old_concentration_solution_neg = old_old_concentration_solution_neg;
        old_old_concentration_solution_neg    = old_concentration_solution_neg;
        old_concentration_solution_neg     = concentration_solution_neg;
        old_old_old_concentration_solution_pos = old_old_concentration_solution_pos;
        old_old_concentration_solution_pos    = old_concentration_solution_pos;
        old_concentration_solution_pos    = concentration_solution_pos;



        double  min_concentration=0;
        double  max_concentration=1;
#ifdef OUTPUT_FILE
        output_file_m << global_concentrational_integrals << std::endl;
        output_file_c << min_concentration << ' ' << max_concentration
                      << std::endl;
#endif

    }
    // Do all the above until we arrive at time 100.
    while (timestep_number <= 20);
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
