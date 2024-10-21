__copyright__ = """
Copyright (C) 2017 Bogdan Enache
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
import os

import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools
from arraycontext import flatten
from meshmode.mesh import BTAG_ALL
from pytools.obj_array import flat_obj_array

import grudge.dof_desc as dof_desc
import grudge.op as op
from grudge.array_context import PyOpenCLArrayContext


logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `weak.py`)

class Plotter:
    def __init__(self, actx, dcoll, order, visualize=True, ylim=None):
        self.actx = actx
        self.dim = dcoll.ambient_dim

        self.visualize = visualize
        if not self.visualize:
            return

        if self.dim == 1:
            import matplotlib.pyplot as pt
            self.fig = pt.figure(figsize=(8, 8), dpi=300)
            self.ylim = ylim

            volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)
            self.x = actx.to_numpy(flatten(volume_discr.nodes()[0], self.actx))
        else:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(dcoll)

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.dim == 1:
            u = self.actx.to_numpy(flatten(evt.state_component, self.actx))

            filename = f"{basename}.png"
            if not overwrite and os.path.exists(filename):
                from meshmode import FileExistsError
                raise FileExistsError(f"output file '{filename}' already exists")

            ax = self.fig.gca()
            ax.plot(self.x, u, "-")
            ax.plot(self.x, u, "k.")
            if self.ylim is not None:
                ax.set_ylim(self.ylim)

            ax.set_xlabel("$x$")
            ax.set_ylabel("$u$")
            ax.set_title(f"t = {evt.t:.2f}")
            self.fig.savefig(filename)
            self.fig.clf()
        else:
            self.vis.write_vtk_file(f"{basename}.vtu", [
                ("u", evt.state_component)
                ], overwrite=overwrite)

# }}}


def main(ctx_factory, dim=2, order=4, use_quad=False, visualize=False,
         flux_type="upwind", tpe=False, quad_order=None, t_final=None,
         nperiods=1., period=2., v_cycle=False, warp=1.0, nel1d=25):

    if quad_order is None:
        quad_order = order if tpe else 2*order + 1

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    # {{{ parameters

    # domain [0, d]^dim
    d = 1.0
    a = -1.0
    l = d - a
    # number of points in each dimension
    npoints = nel1d

    # final time
    final_time = t_final or nperiods * period
    qtag = dof_desc.DISCR_TAG_QUAD if use_quad else None

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh
    from meshmode.mesh import TensorProductElementGroup
    group_cls = TensorProductElementGroup if tpe else None
    mesh = generate_regular_rect_mesh(
            a=(a,)*dim, b=(d,)*dim,
            npoints_per_axis=(npoints,)*dim,
            order=order, group_cls=group_cls)

    from meshmode.discretization.poly_element import (
        QuadratureGroupFactory,
        InterpolatoryEdgeClusteredGroupFactory
    )

    discr_tag_to_group_factory = {
        dof_desc.DISCR_TAG_BASE:
        InterpolatoryEdgeClusteredGroupFactory(order),
        dof_desc.DISCR_TAG_QUAD:
        QuadratureGroupFactory(quad_order)
    }

    from grudge.discretization import make_discretization_collection
    dcoll = make_discretization_collection(
        actx, mesh,
        discr_tag_to_group_factory=discr_tag_to_group_factory
    )

    # }}}

    # {{{ advection operator

    # gaussian parameters

    # Mengaldo test case
    alpha = 41.
    xc = -0.3
    yc = 0.0

    def poly_vel_initializer(xyz_vec, t=0):
        x = xyz_vec[0]
        y = xyz_vec[1]
        actx = x.array_context
        return actx.np.exp(-alpha*((x-xc)**2 + (y-yc)**2))

    def f_halfcircle(x):
        source_center = np.array([d/2, d/2, d/2])[:dim]
        dist = x - source_center
        return (
                (0.5+0.5*actx.np.tanh(500*(-np.dot(dist, dist) + 0.4**2)))
                * (0.5+0.5*actx.np.tanh(500*(dist[0]))))

    def zero_inflow_bc(dtag, t=0):
        dd = dof_desc.as_dofdesc(dtag, qtag)
        return dcoll.discr_from_dd(dd).zeros(actx)

    from grudge.models.advection import VariableCoefficientAdvectionOperator

    x = actx.thaw(dcoll.nodes())

    # velocity field
    def vel_func(t):
        g_t = 1.
        if v_cycle:
            g_t = np.cos(np.pi * t / period)

        if dim == 1:
            c = g_t * x**warp
        else:
            # solid body rotation
            c = flat_obj_array(
                np.pi * g_t * (0. - x[1])**warp,
                np.pi * g_t * (x[0] - 0.)**warp,
                0
            )[:dim]

        return c

    c = vel_func(0)

    adv_operator = VariableCoefficientAdvectionOperator(
        dcoll,
        c,
        inflow_u=lambda t: zero_inflow_bc(BTAG_ALL, t),
        quad_tag=qtag,
        flux_type=flux_type,
        vel_func=vel_func
    )

    # u = f_halfcircle(x)
    u = poly_vel_initializer(x)

    def rhs(t, u):
        return adv_operator.operator(t, u)

    fudge_fac = 0.7 if tpe else .5
    dt = \
        fudge_fac * actx.to_numpy(
            adv_operator.estimate_rk4_timestep(actx, dcoll, fields=u))
    # dt = dt / 20.

    logger.info("Timestep size: %g", dt)
    u_init = actx.to_numpy(op.norm(dcoll, u, 2))
    logger.info("Initial u: %g", u_init)
    u_bound = 100.0 * u_init
    logger.info("Final time: %g", final_time)
    logger.info("Period: %g", period)
    logger.info("Nperiods: %g", nperiods)
    logger.info("Actual nperiods: %g", final_time / period)

    # }}}

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", float(dt), u, rhs)
    plot = Plotter(actx, dcoll, order, visualize=visualize,
            ylim=[-0.1, 1.1])

    step = 0
    save_event = None
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        save_event = event
        if step % 10 == 0:
            norm_u = actx.to_numpy(op.norm(dcoll, event.state_component, 2))
            plot(event, f"fld-var-velocity-{step:04d}")

            logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)
            # NOTE: These are here to ensure the solution is bounded for the
            # time interval specified
            assert norm_u < u_bound

        step += 1

    plot(save_event, f"fld-var-velocity-{step:04d}")
    norm_err = actx.to_numpy(op.norm(dcoll, save_event.state_component - u, 2))
    logger.info("Final error: %g", norm_err)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--warp", default=1, type=float)
    parser.add_argument("--tfinal", type=float)
    parser.add_argument("--nperiods", type=float, default=1.)
    parser.add_argument("--period", type=float, default=2.)
    parser.add_argument("--tpe", action="store_true")
    parser.add_argument("--use-quad", action="store_true")
    parser.add_argument("--quad-order", type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--agitator", action="store_true")
    parser.add_argument("--nel1d", type=int, default=25)
    parser.add_argument("--flux", default="upwind",
            help="'central' or 'upwind'. Run with central to observe aliasing "
            "instability. Add --use-quad to fix that instability.")
    args = parser.parse_args()

    quad_order = args.quad_order
    order = args.order
    if quad_order is None:
        quad_order = order if args.tpe else order + 3

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context, v_cycle=args.agitator,
         dim=args.dim, nperiods=args.nperiods,
         order=order, quad_order=quad_order, nel1d=args.nel1d,
         use_quad=args.use_quad, period=args.period,
         visualize=args.visualize, t_final=args.tfinal,
         flux_type=args.flux, tpe=args.tpe, warp=args.warp)
