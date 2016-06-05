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
        return_value = 3+sin(pi*p(1))*sin(pi*p(0))*cos( time );
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
        return_value = 3+3*sin(pi*p(1))*sin(pi*p(0))*cos( time );
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
        FE_Q<dim>                           concentration_fe;
        DoFHandler<dim>                     concentration_dof_handler;
        ConstraintMatrix                    concentration_constraints;
        SparsityPattern                     concentration_sparsity_pattern;

        SparseMatrix<double>                concentration_mass_matrix;
        SparseMatrix<double>                concentration_laplace_matrix;
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

      EquationData::PotentialRightHandSide<dim> potential_right_hand_side;
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
     // if (rebuild_poisson_matrix)
      {
        poisson_matrix.reinit (poisson_sparsity_pattern);
        std::cout << "   Assembling poisson..." << std::endl;
        /*
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
        */
        MatrixCreator::create_laplace_matrix(potential_dof_handler,
            QGauss<dim>(potential_fe.degree+1),
            poisson_matrix);
        rebuild_poisson_matrix = false;
      //}

      cell = potential_dof_handler.begin_active();
      potential_right_hand_side.set_time(0);
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
 //           cell_rhs(i) += 
   //           (
 //              concentration_values_pos[q_index]-concentration_values_neg[q_index]
     //          +potential_right_hand_side.value(potential_fe_values.quadrature_point (q_index))
       //       )
         //     *
           //   potential_fe_values.shape_value (i, q_index) *potential_fe_values.JxW (q_index);
          }
          poisson_rhs(local_dof_indices[i]) += cell_rhs(i);
        }

      }
      poisson_constraints.condense (poisson_matrix, poisson_rhs); 

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
      time_step = 0.000001;
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
        MatrixCreator::create_laplace_matrix(concentration_dof_handler,
            QGauss<dim>(concentration_fe.degree+1),
            concentration_laplace_matrix);
        concentration_matrix_neg.add(time_step, concentration_laplace_matrix);
        concentration_matrix_pos.add(time_step, concentration_laplace_matrix);
        /*
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
        //                  diffusivity_func.value(concentration_fe_values.quadrature_point (q_index))*
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
        */
      }

      cell = concentration_dof_handler.begin_active();
      EquationData::ConcentrationNegativeRightHandSide<dim> concentration_neg_right_hand_side;
      EquationData::ConcentrationPositiveRightHandSide<dim> concentration_pos_right_hand_side;
      concentration_neg_right_hand_side.set_time(time);
      concentration_pos_right_hand_side.set_time(time);

      for (; cell!=endc; ++cell)
      {
        cell->get_dof_indices (local_dof_indices);
        concentration_fe_values.reinit (cell);
        poisson_fe_values.reinit (cell);
        poisson_fe_values.get_function_gradients(potential_solution, grad_poisson_values);
        concentration_fe_values.get_function_values(exact_concentration_solution_neg, concentration_neg_values);
        concentration_fe_values.get_function_values(exact_concentration_solution_pos, concentration_pos_values);
        cell_rhs_neg = 0;
        cell_rhs_pos = 0;
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
          {
            cell_rhs_neg(i) += 
              concentration_neg_values[q_index]*concentration_fe_values.shape_value(i, q_index)*concentration_fe_values.JxW (q_index)
              +
              time_step*
              (
               (concentration_fe_values.shape_grad (i, q_index) *
                grad_poisson_values[q_index]*
                mobility_func.value(concentration_fe_values.quadrature_point (q_index))
               )*
               concentration_neg_values[q_index]
               +
               concentration_neg_right_hand_side.value(concentration_fe_values.quadrature_point (q_index))
               *concentration_fe_values.shape_value(i, q_index)
              )*concentration_fe_values.JxW (q_index);
            cell_rhs_pos(i) +=
              concentration_pos_values[q_index]*concentration_fe_values.shape_value(i, q_index)*concentration_fe_values.JxW (q_index)
              +
              time_step*
              (
               (-concentration_fe_values.shape_grad (i, q_index) *
                grad_poisson_values[q_index]*
                mobility_func.value(concentration_fe_values.quadrature_point (q_index))
               )*
               concentration_pos_values[q_index]
               +
               concentration_pos_right_hand_side.value(concentration_fe_values.quadrature_point (q_index))
               *concentration_fe_values.shape_value(i, q_index)
              )*concentration_fe_values.JxW (q_index);
          }
          concentration_rhs_neg(local_dof_indices[i]) += cell_rhs_neg(i);
          concentration_rhs_pos(local_dof_indices[i]) += cell_rhs_pos(i);
        }
      }
      concentration_constraints.condense (concentration_matrix_neg, concentration_rhs_neg); 
      concentration_constraints.condense (concentration_matrix_pos, concentration_rhs_pos); 
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






  // @sect4{DriftDiffusionProblem::run}
  //
  template <int dim>
    void DriftDiffusionProblem<dim>::run ()
    {
      GridGenerator::hyper_rectangle(triangulation,
          Point<dim>  (-1,-1),
          Point<dim>  ( 1, 1));
      global_Omega_diameter = GridTools::diameter (triangulation);
      triangulation.refine_global (5); //if want to use global uniform mesh

      setup_dofs();
      setup_boundary_ids();

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
        concentration_constraints.distribute (exact_potential_solution);
        
        EquationData::ExactSolution_concentration_neg<dim> exact_concentration_neg;
        EquationData::ExactSolution_concentration_pos<dim> exact_concentration_pos;
        exact_concentration_neg.set_time(time);
        exact_concentration_pos.set_time(time);

        VectorTools::interpolate (concentration_dof_handler,
            exact_concentration_neg,
            exact_concentration_solution_neg);
        concentration_constraints.distribute (exact_concentration_solution_neg);

        VectorTools::interpolate (concentration_dof_handler,
            exact_concentration_pos,
            exact_concentration_solution_pos);
        concentration_constraints.distribute (exact_concentration_solution_pos);
       
        if (timestep_number == 0 )
        {
          concentration_solution_neg = exact_concentration_solution_neg;
          concentration_solution_pos = exact_concentration_solution_pos;
        }

        assemble_poisson_system ();
        solve_poisson ();
        output_results ();
        
        assemble_concentration_matrix ();
        solve_concentration ();
        //        apply_bound_preserving_limiter();
        time += time_step;
        ++timestep_number;
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
