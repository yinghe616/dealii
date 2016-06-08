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
    const double elementary_charge   = 1;
    const double mobility_neg        = 1;
    const double mobility_pos        = 1;
    const double diffusivity_neg     = 1;
    const double diffusivity_pos     = 1;
    const double permitivity         = 1;
    const double density             = 1;
    const double err_tol = 1e-10;
    //define mobility
    template <int dim>
      class MobilityValues : public Function<dim>
    {
      public:
        MobilityValues () : Function<dim>(1) {}
        virtual double value (const Point<dim>   &p,
            const unsigned int  component = 0) const;
    };
    template <int dim>
      double
      MobilityValues<dim>::value (const Point<dim> &p,
          const unsigned int) const
      {
        return 1;
        return (((p(0)<=1)&&(p(0)>=-1)) ? mobility_neg:mobility_pos);
      }

    //define permitivity
    template <int dim>
      class PermitivityValues : public Function<dim>
    {
      public:
        PermitivityValues () : Function<dim>(1) {}
        virtual double value (const Point<dim>   &p,
            const unsigned int  component = 0) const;
    };
    template <int dim>
      double
      PermitivityValues<dim>::value (const Point<dim> &p,
          const unsigned int) const
      {
        return 1;
        return ((p(0)<=1)&&(p(0)>=-1) ? permitivity:permitivity);
      }

    //define diffusivity
    template <int dim>
      class DiffusivityValues : public Function<dim>
    {
      public:
        DiffusivityValues () : Function<dim>(1) {}
        virtual double value (const Point<dim>   &p,
            const unsigned int  component = 0) const;
    };
    template <int dim>
      double
      DiffusivityValues<dim>::value (const Point<dim> &p,
          const unsigned int) const
      {
        return 1;
        return ((p(0)<=1)&&(p(0)>=-1) ? diffusivity_neg:diffusivity_pos);
      }

    //define initddial values
    template <int dim>
      class ConcentrationNegativeInitialValues : public Function<dim>
    {
      public:
        ConcentrationNegativeInitialValues () : Function<dim>(1) {}

        virtual double value (const Point<dim>   &p,
            const unsigned int  component = 0) const;
    };


    template <int dim>
      double
      ConcentrationNegativeInitialValues<dim>::value (const Point<dim> &p,
          const unsigned int) const
      {
        double return_value;
        const double time = this->get_time ();
        const double pi=3.1415926;
        return_value = 3+cos(pi*p(1))*cos(pi*p(0))*cos( time );
        return return_value;
      }

    template <int dim>
      class ConcentrationPositiveInitialValues : public Function<dim>
    {
      public:
        ConcentrationPositiveInitialValues () : Function<dim>(1) {}

        virtual double value (const Point<dim>   &p,
            const unsigned int  component = 0) const;
    };


    template <int dim>
      double
      ConcentrationPositiveInitialValues<dim>::value (const Point<dim> &p,
          const unsigned int) const
      {
        double return_value;
        const double time = this->get_time ();
        const double pi=3.1415926;
        return_value = 3+3*cos(pi*p(1))*cos(pi*p(0))*cos( time );
        return return_value;
      }

    // define concentration negative right hand side function class
    template <int dim>
      class ConcentrationNegativeRightHandSide : public Function<dim>
    {
      public:
        ConcentrationNegativeRightHandSide () : Function<dim>(1) {}

        virtual double value (const Point<dim>   &p,
            const unsigned int  component = 0) const;
    };


    template <int dim>
      double
      ConcentrationNegativeRightHandSide<dim>::value (const Point<dim>  &p,
          const unsigned int component) const
      {
        const double time = this->get_time ();
        double return_value = 0;
        double pi=3.1415926;
        double f    =  cos(pi*p(0)) * cos(pi*p(1)) * cos (time);
        double dfdt = -cos(pi*p(0)) * cos(pi*p(1)) * sin (time);
        double dfdx = -pi*sin(pi*p(0)) * cos(pi*p(1)) * cos (time);
        double dfdy = -pi*cos(pi*p(0)) * sin(pi*p(1)) * cos (time);
        double dfdx2 = dfdx*dfdx +dfdy*dfdy;
        return_value = dfdt + (2*pi*pi -6) * f - 2*f*f +1/(pi*pi)*dfdx2;
        return return_value;

      }

    // define concentration positive right hand side function class
    template <int dim>
      class ConcentrationPositiveRightHandSide : public Function<dim>
    {
      public:
        ConcentrationPositiveRightHandSide () : Function<dim>(1) {}

        virtual double value (const Point<dim>   &p,
            const unsigned int  component = 0) const;
    };


    template <int dim>
      double
      ConcentrationPositiveRightHandSide<dim>::value (const Point<dim>  &p,
          const unsigned int component) const
      {
        const double time = this->get_time ();
        double return_value = 0;
        double pi=3.1415926;
        double f    =  cos(pi*p(0)) * cos(pi*p(1)) * cos (time);
        double dfdt = -cos(pi*p(0)) * cos(pi*p(1)) * sin (time);
        double dfdx = -pi*sin(pi*p(0)) * cos(pi*p(1)) * cos (time);
        double dfdy = -pi*cos(pi*p(0)) * sin(pi*p(1)) * cos (time);
        double dfdx2 = dfdx*dfdx +dfdy*dfdy;
        return_value = 3*dfdt + (6*pi*pi +6) * f + 6*f*f +3/(pi*pi)*dfdx2;
        return return_value;
      }

    // define potential right hand side function class
    template <int dim>
      class PotentialRightHandSide : public Function<dim>
    {
      public:
        PotentialRightHandSide () : Function<dim>(1) {}

        virtual double value (const Point<dim>   &p,
            const unsigned int  component = 0) const;
    };


    template <int dim>
      double
      PotentialRightHandSide<dim>::value (const Point<dim>  &p,
          const unsigned int component) const
      {
        const double time = this->get_time ();
        double return_value = 0;
        double pi=3.1415926;
        double f    =  cos(pi*p(0)) * cos(pi*p(1)) * cos (time);
        double dfdt = -cos(pi*p(0)) * cos(pi*p(1)) * sin (time);
        double dfdx = -pi*sin(pi*p(0)) * cos(pi*p(1)) * cos (time);
        double dfdy = -pi*cos(pi*p(0)) * sin(pi*p(1)) * cos (time);
        double dfdx2 = dfdx*dfdx +dfdy*dfdy;
        return_value = 2*f;
        //return_value = -4;
        return return_value;
      }

    //define exact solution
    template <int dim>
      class ExactSolution_potential : public Function<dim>
    {
      public:
        ExactSolution_potential () : Function<dim>(1) {}
        virtual double value (const Point<dim>   &p, 
            const unsigned int  component = 0) const;
    };
    template <int dim>
      double
      ExactSolution_potential<dim>::value (const Point<dim> &p, 
          const unsigned int) const
      {
        const double time = this->get_time ();
        double return_value=0;
        double pi=3.1415926;
        return_value = 1/(pi*pi) * cos(pi*p(0)) * cos(pi*p(1)) * cos (time);  
        //return_value = p(0)*p(0)+p(1)*p(1);
        return return_value;
      }

    template <int dim>
      class ExactSolution_concentration_neg : public Function<dim>
    {
      public:
        ExactSolution_concentration_neg () : Function<dim>(1) {}
        virtual double value (const Point<dim>   &p, 
            const unsigned int  component = 0) const;
    };
    template <int dim>
      double
      ExactSolution_concentration_neg<dim>::value (const Point<dim> &p, 
          const unsigned int) const
      {
        const double time = this->get_time ();
        double return_value=0;
        double pi=3.1415926;
        return_value = 3 + cos(pi*p(0)) * cos(pi*p(1)) * cos (time);  
        return return_value;
      }

    template <int dim>
      class ExactSolution_concentration_pos : public Function<dim>
    {
      public:
        ExactSolution_concentration_pos () : Function<dim>(1) {}
        virtual double value (const Point<dim>   &p, 
            const unsigned int  component = 0) const;
    };
    template <int dim>
      double
      ExactSolution_concentration_pos<dim>::value (const Point<dim> &p, 
          const unsigned int) const
      {
        const double time = this->get_time ();
        double return_value=0;
        double pi=3.1415926;
        return_value = 3+ 3*cos(pi*p(0)) * cos(pi*p(1)) * cos (time);  
        return return_value;
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
        Vector<double>                      exact_potential_solution;


        const unsigned int                  concentration_degree;
        FE_DGQ<dim>                         concentration_fe;
        DoFHandler<dim>                     concentration_dof_handler;
        ConstraintMatrix                    concentration_constraints;
        SparsityPattern                     concentration_sparsity_pattern;

        SparseMatrix<double>                concentration_mass_matrix;
        SparseMatrix<double>                concentration_laplace_matrix;
        SparseMatrix<double>                concentration_advec_matrix;
        SparseMatrix<double>                concentration_face_advec_matrix;
        SparseMatrix<double>                concentration_face_diffusion_matrix;
        SparseMatrix<double>                concentration_matrix_neg;
        SparseMatrix<double>                concentration_matrix_pos;


        Vector<double>                      concentration_solution_neg;
        Vector<double>                      exact_concentration_solution_neg;
        Vector<double>                      old_concentration_solution_neg;
        Vector<double>                      old_old_concentration_solution_neg;
        Vector<double>                      old_old_old_concentration_solution_neg;
        Vector<double>                      concentration_rhs_neg;

        Vector<double>                      concentration_solution_pos;
        Vector<double>                      exact_concentration_solution_pos;
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
        //
        // adding advection operators
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
        void integrate_face_term_advection_neg (DoFInfo &dinfo1,
            DoFInfo &dinfo2,
            CellInfo &info1,
            CellInfo &info2,
            Vector<double> &coef);
        void integrate_face_term_advection_pos (DoFInfo &dinfo1,
            DoFInfo &dinfo2,
            CellInfo &info1,
            CellInfo &info2,
            Vector<double> &coef);
        // adding diffusion operators
        void integrate_boundary_term_diffusion_x_u (DoFInfo &dinfo,
            CellInfo &info);
        void integrate_boundary_term_diffusion_y_u (DoFInfo &dinfo,
            CellInfo &info);
        void integrate_boundary_term_diffusion_x_q (DoFInfo &dinfo,
            CellInfo &info);
        void integrate_boundary_term_diffusion_y_q (DoFInfo &dinfo,
            CellInfo &info);
        void integrate_face_term_diffusion_x (DoFInfo &dinfo1,
            DoFInfo &dinfo2,
            CellInfo &info1,
            CellInfo &info2);
        void integrate_face_term_diffusion_y (DoFInfo &dinfo1,
            DoFInfo &dinfo2,
            CellInfo &info1,
            CellInfo &info2);
        void integrate_face_term_diffusion (DoFInfo &dinfo1,
            DoFInfo &dinfo2,
            CellInfo &info1,
            CellInfo &info2,
            Vector<double> &coef);

    };

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
      rebuild_poisson_matrix (true),
      rebuild_concentration_matrices (true)
      {
#ifdef OUTPUT_FILE
        output_file_m.open(output_path_m.c_str());
        output_file_c.open(output_path_c.c_str());
#endif
      }


  // @sect4{The local integrators}

  // These are the functions given to the MeshWorker::integration_loop()
  // called just above. They compute the local contributions to the system
  // matrix and right hand side on cells and faces.
  template <int dim>
    void DriftDiffusionProblem<dim>::integrate_cell_term_advection (DoFInfo &dinfo,
        CellInfo &info,
        Vector<double> &coef)
    { 
      // compute advection term -(c_n\grad\phi,\grad\psi)
      const FEValuesBase<dim> &fe_v = info.fe_values();
      FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
      const std::vector<double> &JxW = fe_v.get_JxW_values ();

      //construct potential_cell and fe_values
      typename DoFHandler<dim>::active_cell_iterator potential_cell(&(dinfo.cell->get_triangulation()),
          dinfo.cell->level(),
          dinfo.cell->index(),
          &potential_dof_handler);
      const QGauss<dim> quadrature_formula(concentration_degree+1);
      const unsigned int n_q_points = quadrature_formula.size();
      FEValues<dim> potential_fe_values(potential_fe,quadrature_formula,update_values | update_gradients);
      potential_fe_values.reinit(potential_cell);

      std::vector<Tensor<1,dim> > potential_grad_values(n_q_points);

      potential_fe_values.get_function_gradients (coef,
          potential_grad_values);

      for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
      {
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        {   for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
          {
            local_matrix(i,j) -= potential_grad_values[point]*fe_v.shape_grad(i,point)*
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
    {
      // First, let us retrieve some of the objects used here from @p info. Note
      // that these objects can handle much more complex structures, thus the
      // access here looks more complicated than might seem necessary.
      const FEValuesBase<dim> &fe_v = info.fe_values();
      FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
      Vector<double> &local_vector = dinfo.vector(0).block(0);
      const std::vector<double> &JxW = fe_v.get_JxW_values ();
      std::vector<double> g(fe_v.n_quadrature_points);
      EquationData::ConcentrationNegativeRightHandSide<dim>  concentration_right_hand_side;
      concentration_right_hand_side.value_list (fe_v.get_quadrature_points(), g);

      typename DoFHandler<dim>::active_cell_iterator potential_cell(&(dinfo.cell->get_triangulation()),
          dinfo.cell->level(),
          dinfo.cell->index(),
          &potential_dof_handler);
      const QGauss<dim> quadrature_formula(concentration_degree+1);
      const unsigned int n_q_points = fe_v.n_quadrature_points;
      FEValues<dim> potential_fe_values(potential_fe,quadrature_formula,update_values| update_gradients);
      potential_fe_values.reinit(potential_cell);


      std::vector<Tensor<1,dim> > potential_grad_values(n_q_points);

      potential_fe_values.get_function_gradients (coef, potential_grad_values);
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

      //construct potential_cell and fe_values
      typename DoFHandler<dim>::active_cell_iterator potential_cell(&(dinfo.cell->get_triangulation()),
          dinfo.cell->level(),
          dinfo.cell->index(),
          &potential_dof_handler);
      const QGauss<dim> quadrature_formula(concentration_degree+1);
      const unsigned int n_q_points = quadrature_formula.size();
      FEValues<dim> potential_fe_values(potential_fe,quadrature_formula,update_values|update_gradients);
      potential_fe_values.reinit(potential_cell);


      std::vector<Tensor<1,dim> > potential_grad_values(n_q_points);

      potential_fe_values.get_function_gradients (coef,
          potential_grad_values);

      for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
      {
        const double beta_n=potential_grad_values[point]* normals[point];
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
    void DriftDiffusionProblem<dim>::integrate_face_term_advection_neg (DoFInfo &dinfo1,
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

      //construct potential_cell and fe_values
      typename DoFHandler<dim>::active_cell_iterator potential_cell1(&(dinfo1.cell->get_triangulation()),
          dinfo1.cell->level(),
          dinfo1.cell->index(),
          &potential_dof_handler);
      typename DoFHandler<dim>::active_cell_iterator potential_cell2(&(dinfo2.cell->get_triangulation()),
          dinfo2.cell->level(),
          dinfo2.cell->index(),
          &potential_dof_handler);

      const QGauss<dim-1> quadrature_formula(concentration_degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEFaceValues<dim> potential_fe_values1(potential_fe,quadrature_formula,update_values| update_gradients);
      potential_fe_values1.reinit(potential_cell1,dinfo1.face_number);

      FEFaceValues<dim> potential_fe_values2(potential_fe,quadrature_formula,update_values| update_gradients);
      potential_fe_values2.reinit(potential_cell2,dinfo2.face_number);

      std::vector<Tensor<1,dim> > potential_grad_values1(n_q_points);
      std::vector<Tensor<1,dim> > potential_grad_values2(n_q_points);

      potential_fe_values1.get_function_gradients (coef,
          potential_grad_values1);
      potential_fe_values2.get_function_gradients (coef,
          potential_grad_values2);

      for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
      {
        double beta_n1= potential_grad_values1[point] * normals[point];
        double beta_n2= potential_grad_values2[point] * normals[point];
        beta_n1 +=beta_n2;
        beta_n1 *=0.5;
        if (beta_n1>=0)
        {
          // This term we've already seen:
          for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
              u1_v1_matrix(i,j) += beta_n1 *
                fe_v.shape_value(j,point) *
                fe_v.shape_value(i,point) *
                JxW[point];

          // We additionally assemble the term $(\beta\cdot n u,\hat
          // v)_{\partial \kappa_+}$,
          for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
            for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
              u1_v2_matrix(k,j) -= beta_n1 *
                fe_v.shape_value(j,point) *
                fe_v_neighbor.shape_value(k,point) *
                JxW[point];
        }
        else
        {
          // This one we've already seen, too:
          for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
              u2_v1_matrix(i,l) += beta_n1 *
                fe_v_neighbor.shape_value(l,point) *
                fe_v.shape_value(i,point) *
                JxW[point];
          // And this is another new one: $(\beta\cdot n \hat u,\hat
          // v)_{\partial \kappa_-}$:
          for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
            for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
              u2_v2_matrix(k,l) -= beta_n1 *
                fe_v_neighbor.shape_value(l,point) *
                fe_v_neighbor.shape_value(k,point) *
                JxW[point];
        }
      }
    }
  
  template <int dim>
    void DriftDiffusionProblem<dim>::integrate_face_term_advection_pos (DoFInfo &dinfo1,
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

      //construct potential_cell and fe_values
      typename DoFHandler<dim>::active_cell_iterator potential_cell1(&(dinfo1.cell->get_triangulation()),
          dinfo1.cell->level(),
          dinfo1.cell->index(),
          &potential_dof_handler);
      typename DoFHandler<dim>::active_cell_iterator potential_cell2(&(dinfo2.cell->get_triangulation()),
          dinfo2.cell->level(),
          dinfo2.cell->index(),
          &potential_dof_handler);

      const QGauss<dim-1> quadrature_formula(concentration_degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEFaceValues<dim> potential_fe_values1(potential_fe,quadrature_formula,update_values| update_gradients);
      potential_fe_values1.reinit(potential_cell1,dinfo1.face_number);

      FEFaceValues<dim> potential_fe_values2(potential_fe,quadrature_formula,update_values| update_gradients);
      potential_fe_values2.reinit(potential_cell2,dinfo2.face_number);

      std::vector<Tensor<1,dim> > potential_grad_values1(n_q_points);
      std::vector<Tensor<1,dim> > potential_grad_values2(n_q_points);

      potential_fe_values1.get_function_gradients (coef,
          potential_grad_values1);
      potential_fe_values2.get_function_gradients (coef,
          potential_grad_values2);

      for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
      {
        double beta_n1= -potential_grad_values1[point] * normals[point];
        double beta_n2= -potential_grad_values2[point] * normals[point];
       // beta_n1 +=beta_n2;
       // beta_n1 *=0.5;
        if (beta_n1>=0)
        {
          // This term we've already seen:
          for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
              u1_v1_matrix(i,j) += beta_n1 *
                fe_v.shape_value(j,point) *
                fe_v.shape_value(i,point) *
                JxW[point];

          // We additionally assemble the term $(\beta\cdot n u,\hat
          // v)_{\partial \kappa_+}$,
          for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
            for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
              u1_v2_matrix(k,j) -= beta_n1 *
                fe_v.shape_value(j,point) *
                fe_v_neighbor.shape_value(k,point) *
                JxW[point];
        }
        else
        {
          // This one we've already seen, too:
          for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
              u2_v1_matrix(i,l) += beta_n1 *
                fe_v_neighbor.shape_value(l,point) *
                fe_v.shape_value(i,point) *
                JxW[point];
          // And this is another new one: $(\beta\cdot n \hat u,\hat
          // v)_{\partial \kappa_-}$:
          for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
            for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
              u2_v2_matrix(k,l) -= beta_n1 *
                fe_v_neighbor.shape_value(l,point) *
                fe_v_neighbor.shape_value(k,point) *
                JxW[point];
        }


      }

    }



  template <int dim>
    void DriftDiffusionProblem<dim>::integrate_face_term_diffusion (DoFInfo &dinfo1,
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

    const QGauss<dim-1> quadrature_formula(concentration_degree+1);
    const unsigned int n_q_points = quadrature_formula.size();

    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
      // assemble the term $-0.5(n_x^- dudx^-, v^-)_{\partial \kappa_-}$,
      for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
          u1_v1_matrix(i,j) -= 0.5*( normals[point] *
            fe_v.shape_grad(j,point) )*
            fe_v.shape_value(i,point) *
            JxW[point];

      // assemble the term $-0.5(n_x^- dudx^+, v^-)_{\partial \kappa_-}$,
      for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
        for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
          u1_v2_matrix(k,j) -= 0.5 * (normals[point] *
            fe_v.shape_grad(j,point)) *
            fe_v_neighbor.shape_value(k,point) *
            JxW[point];

      // assemble the term $0.5(n_x^- dudx^+, v^+)_{\partial \kappa_+}$,
      for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
          u2_v1_matrix(i,l) += 0.5 * (normals[point]*
            fe_v_neighbor.shape_grad(l,point)) *
            fe_v.shape_value(i,point) *
            JxW[point];
      // assemble the term $0.5(n_x^- u^-, v^+)_{\partial \kappa_+}$,
      for (unsigned int k=0; k<fe_v_neighbor.dofs_per_cell; ++k)
        for (unsigned int l=0; l<fe_v_neighbor.dofs_per_cell; ++l)
          u2_v2_matrix(k,l) += 0.5 * (normals[point]*
            fe_v_neighbor.shape_grad(l,point)) *
            fe_v_neighbor.shape_value(k,point) *
            JxW[point];
    }
  }

  template <int dim>
    void DriftDiffusionProblem<dim>::integrate_boundary_term_diffusion_x_q (DoFInfo &dinfo,
        CellInfo &info)
    {
    }

  template <int dim>
    void DriftDiffusionProblem<dim>::integrate_boundary_term_diffusion_y_q (DoFInfo &dinfo,
        CellInfo &info)
    {
    }

  template <int dim>
    void DriftDiffusionProblem<dim>::integrate_boundary_term_diffusion_x_u (DoFInfo &dinfo,
        CellInfo &info)
    {
    }

  template <int dim>
    void DriftDiffusionProblem<dim>::integrate_boundary_term_diffusion_y_u (DoFInfo &dinfo,
        CellInfo &info)
    {
    }

  // @sect4{DriftDiffusionProblem::get_maximal_potential}

  template <int dim>
    double DriftDiffusionProblem<dim>::get_maximal_potential () const
    {
      const QIterated<dim> quadrature_formula (QTrapez<1>(),
          potential_degree+1);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> fe_values (potential_fe, quadrature_formula, 
          update_values|
          update_gradients
          );
      std::vector<double > potential_values(n_q_points);
      std::vector<Tensor<1,dim> > potential_grad(n_q_points);
      double max_potential = 0;

      typename DoFHandler<dim>::active_cell_iterator
        cell = potential_dof_handler.begin_active(),
             endc = potential_dof_handler.end();
      for (; cell!=endc; ++cell)
      {
        fe_values.reinit (cell);
        fe_values.get_function_gradients (potential_solution,
            potential_grad);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          max_potential = std::max (max_potential, potential_grad[q][0]);
          max_potential = std::max (max_potential, potential_grad[q][1]);
        }
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
// DG dones't need constrains matrix
//        concentration_constraints.clear ();
//        DoFTools::make_hanging_node_constraints (concentration_dof_handler,
//            concentration_constraints);
//        concentration_constraints.close ();
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
        << n_u +  n_T + n_T
        << " (" << n_u << '+'<< n_T << '+' << n_T <<')'
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
        concentration_advec_matrix.clear ();
        concentration_face_advec_matrix.clear ();
        concentration_face_diffusion_matrix.clear ();
        concentration_matrix_neg.clear ();
        concentration_matrix_pos.clear ();

        DynamicSparsityPattern dsp2 (concentration_dof_handler.n_dofs(),
            concentration_dof_handler.n_dofs());
        DoFTools::make_flux_sparsity_pattern (concentration_dof_handler, dsp2);

        concentration_sparsity_pattern.copy_from(dsp2);
        concentration_mass_matrix.reinit (concentration_sparsity_pattern);
        concentration_laplace_matrix.reinit (concentration_sparsity_pattern);
        concentration_advec_matrix.reinit (concentration_sparsity_pattern);
        concentration_face_advec_matrix.reinit (concentration_sparsity_pattern);
        concentration_face_diffusion_matrix.reinit (concentration_sparsity_pattern);
        concentration_matrix_neg.reinit (concentration_sparsity_pattern);
        concentration_matrix_pos.reinit (concentration_sparsity_pattern);
      }

      // Lastly, we set the vectors for the Stokes solutions $\mathbf u^{n-1}$
      // and $\mathbf u^{n-2}$, as well as for the concentrations $T^{n}$,
      // $T^{n-1}$ and $T^{n-2}$ (required for time stepping) and all the system
      // right hand sides to their correct sizes and block structure:
      potential_solution.reinit (potential_dof_handler.n_dofs());
      exact_potential_solution.reinit (potential_dof_handler.n_dofs());
      old_potential_solution.reinit (potential_dof_handler.n_dofs());
      poisson_rhs.reinit (potential_dof_handler.n_dofs());

      concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());
      exact_concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());
      old_concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());
      old_old_concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());
      old_old_old_concentration_solution_neg.reinit (concentration_dof_handler.n_dofs());
      concentration_rhs_neg.reinit (concentration_dof_handler.n_dofs());

      concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());
      exact_concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());
      old_concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());
      old_old_concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());
      old_old_old_concentration_solution_pos.reinit (concentration_dof_handler.n_dofs());
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
              if (std::fabs(cell->face(face_number)->center()[0] - (-1)) < 1e-12)
                cell->face(face_number)->set_boundary_id (1);
              if (std::fabs(cell->face(face_number)->center()[0] - (1)) < 1e-12)
                cell->face(face_number)->set_boundary_id (2);
              if (std::fabs(cell->face(face_number)->center()[1] - (1)) < 1e-12)
                cell->face(face_number)->set_boundary_id (3);
              if (std::fabs(cell->face(face_number)->center()[1] - (-1)) < 1e-12)
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


      const unsigned int   dofs_per_cell = potential_fe.dofs_per_cell;
      const unsigned int   n_q_points    = quadrature_formula.size();

      FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
      Vector<double>            cell_rhs (dofs_per_cell);
      std::vector<double>       concentration_values_neg(n_q_points);
      std::vector<double>       concentration_values_pos(n_q_points);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      const EquationData::PermitivityValues<dim>     permitivity_func;

      if (rebuild_poisson_matrix)
      {
        typename DoFHandler<dim>::active_cell_iterator
        cell = potential_dof_handler.begin_active(),
             endc = potential_dof_handler.end();
        poisson_matrix.reinit (poisson_sparsity_pattern);
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
 //               permitivity_func.value(potential_fe_values.quadrature_point (q_index))*
                  potential_fe_values.shape_grad (i, q_index) *
                  potential_fe_values.shape_grad (j, q_index) *
                  potential_fe_values.JxW (q_index); 
              }
              poisson_matrix.add (local_dof_indices[i],
                  local_dof_indices[j],
                  cell_matrix(i,j));
            }
        }
        /*
        MatrixCreator::create_laplace_matrix(potential_dof_handler,
            QGauss<dim>(potential_fe.degree+1),
            poisson_matrix);
        */
  //      rebuild_poisson_matrix = false;
      }

      typename DoFHandler<dim>::active_cell_iterator
        cell = potential_dof_handler.begin_active(),
             endc = potential_dof_handler.end();
      typename DoFHandler<dim>::active_cell_iterator
        cell2 = concentration_dof_handler.begin_active(),
             endc2 = concentration_dof_handler.end();
      poisson_rhs = 0;
      EquationData::PotentialRightHandSide<dim> potential_right_hand_side;
      potential_right_hand_side.set_time(time);
      for (; cell!=endc, cell2!=endc2; ++cell,++cell2)
      {
        cell->get_dof_indices (local_dof_indices);
        potential_fe_values.reinit (cell);
        cell_rhs = 0;
        concentration_fe_values.reinit (cell2);
        concentration_fe_values.get_function_values(exact_concentration_solution_neg, concentration_values_neg);
        concentration_fe_values.get_function_values(exact_concentration_solution_pos, concentration_values_pos);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
          {
            cell_rhs(i) += 
              (
              concentration_values_pos[q_index]-concentration_values_neg[q_index]
          //     +potential_right_hand_side.value(potential_fe_values.quadrature_point (q_index))
              )
              *
              potential_fe_values.shape_value (i, q_index) *potential_fe_values.JxW (q_index);
          }
          poisson_rhs(local_dof_indices[i]) += cell_rhs(i);
        }

      }
      //poisson_constraints.condense (poisson_matrix, poisson_rhs); 

      // apply boundary conditions
      EquationData::ExactSolution_potential<dim> exact_potential;
      exact_potential.set_time(time);
      std::map<types::global_dof_index,double> boundary_values;
      VectorTools::interpolate_boundary_values (potential_dof_handler,
          1,
          exact_potential,
          boundary_values);
      VectorTools::interpolate_boundary_values (potential_dof_handler,
          2,
          exact_potential,
          boundary_values);
      VectorTools::interpolate_boundary_values (potential_dof_handler,
          3,
          exact_potential,
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
      {
        old_time_step = time_step;
        const double maximal_potential = get_maximal_potential();

#if 0
        std::cout << "max eletric field " << maximal_potential << std::endl;
        double h_min=GridTools::minimal_cell_diameter(triangulation);
        std::cout << "min cell diameter " << h_min << std::endl;
        time_step = 1*h_min*h_min;///maximal_potential;
        std::cout << "Time step " << time_step << std::endl;
#else
        time_step = 0.000001;
#endif
      }

      // set up necessary dofs

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

      // build mass matrix and laplace and advection matrices
      {
        if (rebuild_concentration_matrices)
        {
          std::cout << "   Assembling concentration..." << std::endl;
          concentration_matrix_neg=0;
          concentration_matrix_pos=0;
          //Mass
          MatrixCreator::create_mass_matrix(concentration_dof_handler,
              QGauss<dim>(concentration_fe.degree+1),
              concentration_mass_matrix);
          concentration_matrix_neg.copy_from(concentration_mass_matrix);
          concentration_matrix_pos.copy_from(concentration_mass_matrix);
          //laplace
          MatrixCreator::create_laplace_matrix(concentration_dof_handler,
              QGauss<dim>(concentration_fe.degree+1),
              concentration_laplace_matrix);
          concentration_matrix_neg.add(time_step, concentration_laplace_matrix);
          concentration_matrix_pos.add(time_step, concentration_laplace_matrix);
          // advection 
          FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
          std::vector< Tensor<1,dim> >       grad_poisson_values(n_q_points);
          std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
          const EquationData::MobilityValues<dim>     mobility_func;
          typename DoFHandler<dim>::active_cell_iterator
            cell = concentration_dof_handler.begin_active(),
                 endc = concentration_dof_handler.end();
          typename DoFHandler<dim>::active_cell_iterator
            cell2 = potential_dof_handler.begin_active(),
                 endc2 = potential_dof_handler.end();

          for (; cell!=endc, cell2!=endc2; ++cell, ++cell2)
          { 
            cell_matrix = 0;
            cell->get_dof_indices (local_dof_indices);
            concentration_fe_values.reinit (cell);
            poisson_fe_values.reinit (cell2);
            poisson_fe_values.get_function_gradients(potential_solution, grad_poisson_values);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {  for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
                for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
                { 
                  cell_matrix(i,j) +=
                    (
                     concentration_fe_values.shape_value (j, q_index) *
                     grad_poisson_values[q_index]*
                     concentration_fe_values.shape_grad (i, q_index) *
                     mobility_func.value(concentration_fe_values.quadrature_point (q_index))
                     )*
                    concentration_fe_values.JxW (q_index); 
                }
                concentration_advec_matrix.add (local_dof_indices[i],
                    local_dof_indices[j],
                    cell_matrix(i,j));
              }
            }
          }
          concentration_matrix_neg.add(-time_step, concentration_advec_matrix);
          concentration_matrix_pos.add(+time_step, concentration_advec_matrix);
        }
      }// end of  build mass matrix and laplace and advection matrices

      // build face advection matrix and advection matrices
      {
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
        // obtain advection matrix
        concentration_advec_matrix.reinit (concentration_sparsity_pattern);
        rhs_tmp.reinit(concentration_dof_handler.n_dofs());
        MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> > assembler1;
        assembler1.initialize(concentration_face_advec_matrix, rhs_tmp);
        //bind definitions
        auto integrate_cell_term_advection_bind = std::bind(&DriftDiffusionProblem<dim>::integrate_cell_term_advection,
            this, std::placeholders::_1, std::placeholders::_2, this->exact_potential_solution);
        auto integrate_boundary_term_advection_bind = std::bind(&DriftDiffusionProblem<dim>::integrate_boundary_term_advection,
            this, std::placeholders::_1, std::placeholders::_2, this->exact_potential_solution);
        auto integrate_face_term_advection_neg_bind = std::bind(&DriftDiffusionProblem<dim>::integrate_face_term_advection_neg,
            this, std::placeholders::_1, std::placeholders::_2,std::placeholders::_3, std::placeholders::_4, this->potential_solution);

        MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
          (concentration_dof_handler.begin_active(), concentration_dof_handler.end(),
           dof_info, info_box,
          // integrate_cell_term_advection_bind,
         //  integrate_boundary_term_advection_bind,
           NULL,
           NULL,
           integrate_face_term_advection_neg_bind,
           assembler1);

        concentration_matrix_neg.add(+time_step, concentration_face_advec_matrix);
        
        concentration_face_advec_matrix.reinit (concentration_sparsity_pattern);
        rhs_tmp.reinit(concentration_dof_handler.n_dofs());
        MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> > assembler3;
        assembler3.initialize(concentration_face_advec_matrix, rhs_tmp);
        auto integrate_face_term_advection_pos_bind = std::bind(&DriftDiffusionProblem<dim>::integrate_face_term_advection_pos,
            this, std::placeholders::_1, std::placeholders::_2,std::placeholders::_3, std::placeholders::_4, this->potential_solution);
        MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
          (concentration_dof_handler.begin_active(), concentration_dof_handler.end(),
           dof_info, info_box,
           NULL,
           NULL,
           integrate_face_term_advection_pos_bind,
           assembler3);

        concentration_matrix_pos.add(+time_step, concentration_face_advec_matrix);

        // build face diffusion matrix matrices
        if (rebuild_concentration_matrices)
        {
          concentration_face_diffusion_matrix.reinit (concentration_sparsity_pattern);
          rhs_tmp.reinit(concentration_dof_handler.n_dofs());
       
          MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, Vector<double> > assembler2;
       
          assembler2.initialize(concentration_face_diffusion_matrix, rhs_tmp);
        
          //bind definitions
          auto integrate_face_term_diffusion_bind = std::bind(&DriftDiffusionProblem<dim>::integrate_face_term_diffusion,
              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, this->potential_solution);

          MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
            (concentration_dof_handler.begin_active(), concentration_dof_handler.end(),
             dof_info, info_box,
             NULL,
             NULL,
             integrate_face_term_diffusion_bind,
             assembler2);

          concentration_matrix_neg.add(-time_step, concentration_face_diffusion_matrix);
          concentration_matrix_pos.add(-time_step, concentration_face_diffusion_matrix);
          rebuild_concentration_matrices = false;
        }

      }
      // build right hand side vector
      {
        concentration_rhs_neg=0;
     //   concentration_advec_matrix.vmult(concentration_rhs_neg, concentration_solution_neg);
        concentration_rhs_neg*= -time_step;
        Vector<double> sol_tmp (concentration_dof_handler.n_dofs());
        concentration_mass_matrix.vmult(sol_tmp, concentration_solution_neg);
        concentration_rhs_neg += sol_tmp;

        concentration_rhs_pos=0;
     //   concentration_advec_matrix.vmult(concentration_rhs_pos, concentration_solution_pos);
        concentration_rhs_pos *= +time_step;
        concentration_mass_matrix.vmult(sol_tmp, concentration_solution_pos);
        concentration_rhs_pos += sol_tmp;
      }
      // DG elements dont need to apply the constraints
      //      concentration_constraints.condense (concentration_matrix_neg, concentration_rhs_neg); 
      //      concentration_constraints.condense (concentration_matrix_pos, concentration_rhs_pos); 
      /*

         EquationData::ExactSolution_concentration_neg<dim> exact_concentration_neg;
         EquationData::ExactSolution_concentration_pos<dim> exact_concentration_pos;
         exact_concentration_neg.set_time(time);
         exact_concentration_pos.set_time(time);
         std::map<types::global_dof_index,double> boundary_values_neg;
         VectorTools::interpolate_boundary_values (concentration_dof_handler,
         1,
         exact_concentration_neg,
         boundary_values_neg);
         VectorTools::interpolate_boundary_values (concentration_dof_handler,
         2,
         exact_concentration_neg,
         boundary_values_neg);
         VectorTools::interpolate_boundary_values (concentration_dof_handler,
         3,
         exact_concentration_neg,
         boundary_values_neg);
         MatrixTools::apply_boundary_values (boundary_values_neg,
         concentration_matrix_neg,
         concentration_solution_neg,
         concentration_rhs_neg);

         std::map<types::global_dof_index,double> boundary_values_pos;
         VectorTools::interpolate_boundary_values (concentration_dof_handler,
         1,
         exact_concentration_pos,
         boundary_values_pos);
         VectorTools::interpolate_boundary_values (concentration_dof_handler,
         2,
         exact_concentration_pos,
         boundary_values_pos);
         VectorTools::interpolate_boundary_values (concentration_dof_handler,
         3,
         exact_concentration_pos,
         boundary_values_pos);
         MatrixTools::apply_boundary_values (boundary_values_pos,
         concentration_matrix_pos,
         concentration_solution_pos,
         concentration_rhs_pos);
         */
    }



  // @sect4{DriftDiffusionProblem::solve}
  //
  template <int dim>
    void DriftDiffusionProblem<dim>::solve_poisson ()
    {
      std::cout << "   Solving poisson..." << std::endl;
      {
        SolverControl solver_control (poisson_matrix.m(),
            1e-8*poisson_rhs.l2_norm());
      //  SolverControl solver_control (1000, 1e-12);
        SolverCG<> cg(solver_control);
        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(poisson_matrix, 1.0);
        
        //potential_solution=0;

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
        //SolverControl solver_control (concentration_matrix_neg.m(),
          //  1e-6*concentration_rhs_neg.l2_norm());
        SolverControl solver_control (1000, 1e-12);
        SolverGMRES<> gmres(solver_control);
        PreconditionSSOR<> preconditioner_neg;
        preconditioner_neg.initialize(concentration_matrix_neg, 1.0);

        concentration_solution_neg=0;
        gmres.solve(concentration_matrix_neg, concentration_solution_neg, concentration_rhs_neg,
           preconditioner_neg);

       // concentration_constraints.distribute (concentration_solution_neg);

        std::cout << "   "
          << solver_control.last_step()
          << "GMRES iterations for concentration equation."
          << std::endl;

        //SolverControl solver_control_2 (concentration_matrix_pos.m(),
          //  1e-6*concentration_rhs_pos.l2_norm());
        SolverControl solver_control_2 (1000, 1e-12);
        SolverGMRES<> gmres_2(solver_control_2);
        PreconditionSSOR<> preconditioner_pos;
        preconditioner_pos.initialize(concentration_matrix_pos, 1.0);

        concentration_solution_pos=0;
        gmres_2.solve(concentration_matrix_pos, concentration_solution_pos, concentration_rhs_pos, 
            preconditioner_pos);

        //concentration_constraints.distribute (concentration_solution_pos);

        std::cout << "   "
          << solver_control_2.last_step()
          << "GMRES iterations for concentration equation."
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
      data_out.add_data_vector (potential_dof_handler, exact_potential_solution,
          "Exact_Potential");
      data_out.add_data_vector (concentration_dof_handler, exact_concentration_solution_neg,
          "exact_Concentration_negative");
      data_out.add_data_vector (concentration_dof_handler, exact_concentration_solution_pos,
          "exact_Concentration_positive");
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
      estimated_error_per_cell =0;
      Vector<double> tmp_diff;
      tmp_diff  = concentration_solution_pos;
      tmp_diff -= concentration_solution_neg;
   /*
    KellyErrorEstimator<dim>::estimate (concentration_dof_handler,
          QGauss<dim-1>(concentration_degree+1),
          typename FunctionMap<dim>::type(),
          concentration_solution_neg,
          estimated_error_per_cell);

      estimated_error_per_cell +=estimated_error_per_cell_2;
      Vector<float> estimated_error_per_cell_3 (triangulation.n_active_cells());
      KellyErrorEstimator<dim>::estimate (potential_dof_handler,
          QGauss<dim-1>(potential_degree+1),
          typename FunctionMap<dim>::type(),
          potential_solution,
          estimated_error_per_cell_3);
    */  
      Vector<float> estimated_error_per_cell_2 (triangulation.n_active_cells());
      KellyErrorEstimator<dim>::estimate (concentration_dof_handler,
          QGauss<dim-1>(concentration_degree+1),
          typename FunctionMap<dim>::type(),
          //concentration_solution_pos,
          tmp_diff,
          estimated_error_per_cell_2);
      estimated_error_per_cell +=estimated_error_per_cell_2;
#if 1
      GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
          estimated_error_per_cell,
          .3, 0.00);
#else
      GridRefinement::refine(triangulation,
          estimated_error_per_cell,
          .6);

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
      std::vector<Vector<double>> x_concentration (8);
      x_concentration[0] = concentration_solution_neg;
      x_concentration[1] = exact_concentration_solution_neg;
      x_concentration[2] = old_concentration_solution_neg;
      x_concentration[3] = old_old_concentration_solution_neg;
      x_concentration[4] = concentration_solution_pos;
      x_concentration[5] = exact_concentration_solution_pos;
      x_concentration[6] = old_concentration_solution_pos;
      x_concentration[7] = old_old_concentration_solution_pos;
      std::vector<Vector<double>> x_potential (2);
      x_potential[0] = potential_solution;
      x_potential[1] = exact_potential_solution;

      SolutionTransfer<dim,Vector<double>>
        concentration_trans(concentration_dof_handler);
      SolutionTransfer<dim,Vector<double>>
        poisson_trans(potential_dof_handler);

      triangulation.prepare_coarsening_and_refinement();
      concentration_trans.prepare_for_coarsening_and_refinement(x_concentration);
      poisson_trans.prepare_for_coarsening_and_refinement(x_potential);

      // Now everything is ready, so do the refinement and recreate the dof
      // structure on the new grid, and initialize the matrix structures and the
      triangulation.execute_coarsening_and_refinement ();
      setup_dofs ();
      setup_boundary_ids();

      std::vector<Vector<double>> tmp (8);
      tmp[0].reinit (concentration_solution_neg);
      tmp[1].reinit (concentration_solution_neg);
      tmp[2].reinit (concentration_solution_neg);
      tmp[3].reinit (concentration_solution_neg);
      tmp[4].reinit (concentration_solution_neg);
      tmp[5].reinit (concentration_solution_neg);
      tmp[6].reinit (concentration_solution_neg);
      tmp[7].reinit (concentration_solution_neg);
      concentration_trans.interpolate(x_concentration, tmp);

      concentration_solution_neg         = tmp[0];
      exact_concentration_solution_neg   = tmp[1];
      old_concentration_solution_neg     = tmp[2];
      old_old_concentration_solution_neg = tmp[3];
      concentration_solution_pos         = tmp[4];
      exact_concentration_solution_pos   = tmp[5];
      old_concentration_solution_pos     = tmp[6];
      old_old_concentration_solution_pos = tmp[7];
// dg don't need to consider hanging nodes constrains
//      concentration_constraints.distribute (concentration_solution_neg);
//      concentration_constraints.distribute (exact_concentration_solution_neg);
//      concentration_constraints.distribute (concentration_solution_pos);
//      concentration_constraints.distribute (exact_concentration_solution_pos);

      std::vector<Vector<double>> tmp2 (2);
      tmp2[0].reinit (potential_solution);
      tmp2[1].reinit (potential_solution);
      poisson_trans.interpolate (x_potential, tmp2);
      potential_solution         = tmp2[0];
      exact_potential_solution   = tmp2[1];
      poisson_constraints.distribute (potential_solution);
      poisson_constraints.distribute (exact_potential_solution);

      rebuild_poisson_matrix          = true;
      rebuild_concentration_matrices  = true;
    }




  // @sect4{DriftDiffusionProblem::run}
  //
  template <int dim>
    void DriftDiffusionProblem<dim>::run ()
    {
      const unsigned int initial_refinement = (dim == 2 ? 4 : 2);
      const unsigned int n_pre_refinement_steps = (dim == 2 ? 2 : 3);

      GridGenerator::hyper_rectangle(triangulation,
          Point<dim>  (-1,-1),
          Point<dim>  ( 1, 1));
      global_Omega_diameter = GridTools::diameter (triangulation);
      triangulation.refine_global (5); //if want to use global uniform mesh
  //    triangulation.refine_global (initial_refinement);

      setup_dofs();
      setup_boundary_ids();
      unsigned int pre_refinement_step = 0;

start_time_iteration:

      time =0;
      timestep_number           = 0;
      time_step = old_time_step = 0;

      do
      {
        std::cout << "Timestep " << timestep_number
          << ":  t=" << time
          << std::endl;
//set up exact solutions
        EquationData::ExactSolution_potential<dim> exact_potential;
        exact_potential.set_time(time);
        VectorTools::interpolate (potential_dof_handler,
            exact_potential,
            exact_potential_solution);
        poisson_constraints.distribute (exact_potential_solution);
        
        EquationData::ExactSolution_concentration_neg<dim> exact_concentration_neg;
        exact_concentration_neg.set_time(time);
        VectorTools::interpolate (concentration_dof_handler,
            exact_concentration_neg,
            exact_concentration_solution_neg);
//        concentration_constraints.distribute (exact_concentration_solution_neg);

        EquationData::ExactSolution_concentration_pos<dim> exact_concentration_pos;
        exact_concentration_pos.set_time(time);
        VectorTools::interpolate (concentration_dof_handler,
            exact_concentration_pos,
            exact_concentration_solution_pos);
//        concentration_constraints.distribute (exact_concentration_solution_pos);
       
        if (timestep_number == 0 )
        {
          concentration_solution_neg = exact_concentration_solution_neg;
          concentration_solution_pos = exact_concentration_solution_pos;
        }

        assemble_poisson_system ();
        solve_poisson ();
        
        assemble_concentration_matrix ();
        solve_concentration ();
        //        apply_bound_preserving_limiter();
        //
        // adaptive mesh refinement
#ifdef AMR
        if ((timestep_number == 0) &&
                (pre_refinement_step < n_pre_refinement_steps))
        {
            refine_mesh (initial_refinement + n_pre_refinement_steps);
          //  apply_bound_preserving_limiter();
            ++pre_refinement_step;
            timestep_number=pre_refinement_step*100+1;
            output_results ();
            timestep_number =0;
            goto start_time_iteration;
        }
        else if ((timestep_number > 0) && (timestep_number % 1 == 0))
        {
            refine_mesh (initial_refinement + n_pre_refinement_steps);
            //apply_bound_preserving_limiter();
        }

#endif

        //
        time += time_step;
        ++timestep_number;
        output_results ();
        old_potential_solution             = potential_solution;
        old_potential_solution            -= exact_potential_solution;
        old_concentration_solution_neg     = concentration_solution_neg;
        old_concentration_solution_neg    -= exact_concentration_solution_neg;
        old_concentration_solution_pos     = concentration_solution_pos;
        old_concentration_solution_pos    -= exact_concentration_solution_pos;
        std::cout << "xxxxx1  " << old_potential_solution.linfty_norm()<<std::endl;
        std::cout << "xxxxx2  " << old_concentration_solution_neg.linfty_norm()<<std::endl;
        std::cout << "xxxxx3  " << old_concentration_solution_pos.linfty_norm()<<std::endl;
/*
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
*/
      }
      // Do all the above until we arrive at time 100.
      while (timestep_number <= 100);
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
