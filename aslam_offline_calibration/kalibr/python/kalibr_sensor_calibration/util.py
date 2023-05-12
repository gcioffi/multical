import colorsys
import matplotlib.patches as patches
import pylab as pl
import sys
from matplotlib.backends.backend_pdf import PdfPages
try:
    # Python 2
    from cStringIO import StringIO
except ImportError:
    # Python 3
    from io import StringIO

import numpy as np
import open3d as o3d
from sm import PlotCollection

from . import plots


def printErrorStatistics(cself, dest=sys.stdout):
    # Reprojection errors
    print("Normalized Residuals\n----------------------------", file=dest)
    try:
        for lidarIdx, lidar in enumerate(cself.LiDARList):
            e2 = np.array(
                [np.sqrt(e.evaluateError()) for p in lidar.targetObs for e in
                 p.errorTerms])
            if e2.size:
                print("Distance error (LiDAR{0}):     count: {1}, mean {2}, median {3}, std: {4}".format(
                    lidarIdx, e2.shape[0], np.mean(e2), np.median(e2), np.std(e2) ), file=dest)
    except:
        pass

    for cidx, cam in enumerate(cself.CameraChain.camList):
        if len(cam.allReprojectionErrors) > 0:
            e2 = np.array([np.sqrt(rerr.evaluateError()) for reprojectionErrors in cam.allReprojectionErrors for rerr in
                           reprojectionErrors])
            print("Reprojection error (cam{0}):     mean {1}, median {2}, std: {3}".format(cidx, np.mean(e2), np.median(e2), np.std(e2) ), file=dest)
        else:
            print("Reprojection error (cam{0}):     no corners".format(cidx), file=dest)

    for iidx, imu in enumerate(cself.ImuList):
        # Gyro errors
        e2 = np.array([np.sqrt(e.evaluateError()) for e in imu.gyroErrors])
        print("Gyroscope error (imu{0}):        mean {1}, median {2}, std: {3}".format(iidx, np.mean(e2), np.median(e2), np.std(e2)), file=dest)
        # Accelerometer errors
        e2 = np.array([np.sqrt(e.evaluateError()) for e in imu.accelErrors])
        print("Accelerometer error (imu{0}):    mean {1}, median {2}, std: {3}".format(iidx, np.mean(e2), np.median(e2), np.std(e2)), file=dest)

    print("", file=dest)
    print("Residuals\n----------------------------", file=dest)
    try:
        for lidarIdx, lidar in enumerate(cself.LiDARList):
            e2 = np.array([np.linalg.norm(e.error()) for p in lidar.targetObs for e in p.errorTerms])
            if e2.size:
                print("Distance error (LiDAR{0}):     count: {1}, mean {2}, median {3}, std: {4}".format(
                    lidarIdx, e2.shape[0], np.mean(e2), np.median(e2), np.std(e2) ), file=dest)
    except:
        pass

    for cidx, cam in enumerate(cself.CameraChain.camList):
        if len(cam.allReprojectionErrors) > 0:
            e2 = np.array([np.linalg.norm(rerr.error()) for reprojectionErrors in cam.allReprojectionErrors for rerr in
                           reprojectionErrors])
            print("Reprojection error (cam{0}) [px]:     mean {1}, median {2}, std: {3}".format(cidx, np.mean(e2), np.median(e2), np.std(e2) ), file=dest)
        else:
            print("Reprojection error (cam{0}) [px]:     no corners".format(cidx), file=dest)

    for iidx, imu in enumerate(cself.ImuList):
        # Gyro errors
        e2 = np.array([np.linalg.norm(e.error()) for e in imu.gyroErrors])
        print("Gyroscope error (imu{0}) [rad/s]:     mean {1}, median {2}, std: {3}".format(iidx, np.mean(e2), np.median(e2), np.std(e2)), file=dest)
        # Accelerometer errors
        e2 = np.array([np.linalg.norm(e.error()) for e in imu.accelErrors])
        print("Accelerometer error (imu{0}) [m/s^2]: mean {1}, median {2}, std: {3}".format(iidx, np.mean(e2), np.median(e2), np.std(e2)), file=dest)


def printGravity(cself):
    print
    print("Gravity vector: (in target coordinates): [m/s^2]")
    print( cself.gravityDv.toEuclidean())


def printResults(cself, withCov=False):
    for tartgetId, target in enumerate(cself.targets):
        print()
        print( "Transformation T_target{0}_world (world to target{0}, T_t_w): ".format(tartgetId))
        print( target.getResultTrafoWorldToTarget())

    reference_sensor = cself.reference_sensor
    try:
        for lidarNr, lidar in enumerate(cself.LiDARList):
            print()
            print( "Transformation T_lidar{0}_{1} ({1} to lidar{0}, T_l_b): ".format(lidarNr, reference_sensor))
            print( lidar.getTransformationReferenceToLiDAR().T())

            if not cself.noTimeCalibration:
                print()
                print( "lidar{0} to {1} time: [s] (t_{1} = t_lidar + shift)".format(lidarNr, reference_sensor))
                print( lidar.lidarOffsetDv.toScalar())
                print()
    except:
        pass

    nCams = len(cself.CameraChain.camList)
    for camNr in range(0, nCams):
        T_cam_ref = cself.CameraChain.getTransformationReferenceToCam(camNr)

        print()
        print( "Transformation T_cam{0}_{1} ({1} to cam{0}, T_c_b): ".format(camNr, reference_sensor))
        if withCov and camNr == 0:
            print( "\t quaternion: ", T_cam_ref.q(), " +- ", cself.std_trafo_ic[0:3])
            print( "\t translation: ", T_cam_ref.t(), " +- ", cself.std_trafo_ic[3:])
        print( T_cam_ref.T())

        if not cself.noTimeCalibration:
            print()
            print( "cam{0} to {1} time: [s] (t_{1} = t_cam + shift)".format(camNr, reference_sensor))
            print( cself.CameraChain.getResultTimeShift(camNr),)

            if withCov:
                print( " +- ", cself.std_times[camNr])
            else:
                print()

    print()
    for (imuNr, imu) in enumerate(cself.ImuList):
        print( "IMU{0}:\n".format(imuNr), "----------------------------")
        imu.getImuConfig().printDetails()


def printBaselines(self):
    # print all baselines in the camera chain
    if nCams > 1:
        for camNr in range(0, nCams - 1):
            T, baseline = cself.CameraChain.getResultBaseline(camNr, camNr + 1)

            if cself.CameraChain.camList[camNr + 1].T_extrinsic_fixed:
                isFixed = "(fixed to external data)"
            else:
                isFixed = ""

            print()
            print( "Baseline (cam{0} to cam{1}): [m] {2}".format(camNr, camNr + 1, isFixed))
            print( T.T())
            print( baseline, "[m]")


def generateReport(cself, filename="report.pdf", showOnScreen=True):
    figs = list()
    plotter = PlotCollection.PlotCollection("Calibration report")
    offset = 3010

    # Output calibration results in text form.
    sstream = StringIO.StringIO()
    printResultTxt(cself, sstream)

    text = [line for line in StringIO.StringIO(sstream.getvalue())]
    linesPerPage = 40

    while True:
        fig = pl.figure(offset)
        offset += 1

        left, width = .05, 1.
        bottom, height = -.05, 1.
        right = left + width
        top = bottom + height

        ax = fig.add_axes([.0, .0, 1., 1.])
        # axes coordinates are 0,0 is bottom left and 1,1 is upper right
        p = patches.Rectangle((left, bottom), width, height, fill=False, transform=ax.transAxes, \
                              clip_on=False, edgecolor="none")
        ax.add_patch(p)
        pl.axis('off')

        printText = lambda t: ax.text(left, top, t, fontsize=8, \
                                      horizontalalignment='left', verticalalignment='top', \
                                      transform=ax.transAxes)

        if len(text) > linesPerPage:
            printText("".join(text[0:linesPerPage]))
            figs.append(fig)
            text = text[linesPerPage:]
        else:
            printText("".join(text[0:]))
            figs.append(fig)
            break

    # plot imu stuff (if we have imus)
    for iidx, imu in enumerate(cself.ImuList):
        f = pl.figure(offset + iidx)
        plots.plotAccelerations(cself, iidx, fno=f.number, noShow=True)
        plotter.add_figure("imu{0}: accelerations".format(iidx), f)
        figs.append(f)
        offset += len(cself.ImuList)

        f = pl.figure(offset + iidx)
        plots.plotAccelErrorPerAxis(cself, iidx, fno=f.number, noShow=True)
        plotter.add_figure("imu{0}: acceleration error".format(iidx), f)
        figs.append(f)
        offset += len(cself.ImuList)

        if not imu.staticBias:
            f = pl.figure(offset + iidx)
            plots.plotAccelBias(cself, iidx, fno=f.number, noShow=True)
            plotter.add_figure("imu{0}: accelerometer bias".format(iidx), f)
            figs.append(f)
            offset += len(cself.ImuList)

        f = pl.figure(offset + iidx)
        plots.plotAngularVelocities(cself, iidx, fno=f.number, noShow=True)
        plotter.add_figure("imu{0}: angular velocities".format(iidx), f)
        figs.append(f)
        offset += len(cself.ImuList)

        f = pl.figure(offset + iidx)
        plots.plotGyroErrorPerAxis(cself, iidx, fno=f.number, noShow=True)
        plotter.add_figure("imu{0}: angular velocity error".format(iidx), f)
        figs.append(f)
        offset += len(cself.ImuList)

        if not imu.staticBias:
            f = pl.figure(offset + iidx)
            plots.plotAngularVelocityBias(cself, iidx, fno=f.number, noShow=True)
            plotter.add_figure("imu{0}: gyroscope bias".format(iidx), f)
            figs.append(f)
            offset += len(cself.ImuList)

    # plot cam stuff
    if cself.CameraChain:
        for cidx, cam in enumerate(cself.CameraChain.camList):
            f = pl.figure(offset + cidx)
            title = "cam{0}: reprojection errors".format(cidx);
            plots.plotReprojectionScatter(cself, cidx, fno=f.number, noShow=True, title=title)
            plotter.add_figure(title, f)
            figs.append(f)
            offset += len(cself.CameraChain.camList)

    # write to pdf
    pdf = PdfPages(filename)
    for fig in figs:
        pdf.savefig(fig)
    pdf.close()

    if showOnScreen:
        plotter.show()


def saveResultTxt(cself, filename='cam_imu_result.txt'):
    f = open(filename, 'w')
    printResultTxt(cself, stream=f)


def printResultTxt(cself, stream=sys.stdout):
    print("Calibration results", file=stream)
    print("===================", file=stream) 
    printErrorStatistics(cself, stream)
    print("", file=stream) 

    reference_sensor = cself.reference_sensor
    for tartgetId, target in enumerate(cself.targets):
        print("", file=stream)
        print("Transformation world to target{0}: ".format(tartgetId), file=stream)
        print(target.getResultTrafoWorldToTarget(), file=stream)
        print("", file=stream)

    try:
        for lidarNr, lidar in enumerate(cself.LiDARList):
            print("", file=stream)
            print("Transformation {1} to lidar{0}: ".format(lidarNr, reference_sensor), file=stream)
            print(lidar.getTransformationReferenceToLiDAR().T(), file=stream)

            if not cself.noTimeCalibration:
                print("", file=stream)
                print("lidar{0} to {1} time: [s] (t_{1} = t_lidar + shift)".format(lidarNr, reference_sensor), file=stream)
                print(lidar.lidarOffsetDv.toScalar(), file=stream)
                print("", file=stream)
    except:
        pass

    # Calibration results
    nCams = len(cself.CameraChain.camList)
    for camNr in range(0, nCams):
        T = cself.CameraChain.getTransformationReferenceToCam(camNr)
        print("", file=stream)
        print("Transformation (cam{0}):".format(camNr), file=stream)
        print("-----------------------", file=stream)
        print("T_cam{0}_{1}:  ({1} to cam{0}): ".format(camNr, reference_sensor), file=stream)
        print(T.T(), file=stream)
        print("", file=stream)
        print("T_{1}_cam{0}:  (cam{0} to {1}): ".format(camNr, reference_sensor), file=stream)
        print(T.inverse().T(), file=stream)

        # Time
        print("", file=stream)
        print("timeshift cam{0} to {1}: [s] (t_imu = t_cam + shift)".format(camNr, reference_sensor), file=stream)
        print(cself.CameraChain.getResultTimeShift(camNr), file=stream)
        print("", file=stream)

    # print all baselines in the camera chain
    if nCams > 1:
        print("Baselines:", file=stream)
        print("----------", file=stream)

        for camNr in range(0, nCams - 1):
            T, baseline = cself.CameraChain.getResultBaseline(camNr, camNr + 1)
            print("Baseline (cam{0} to cam{1}): ".format(camNr, camNr + 1), file=stream)
            print(T.T(), file=stream)
            print("baseline norm: ", baseline, "[m]", file=stream)
            print("", file=stream)

    if cself.ImuList:
        # Gravity
        print("", file=stream)
        print("Gravity vector in target coords: [m/s^2]", file=stream)
        print(cself.gravityDv.toEuclidean(), file=stream)

    print("", file=stream)
    print("", file=stream)
    print("Calibration configuration", file=stream)
    print("=========================", file=stream)
    print("", file=stream)

    for camNr, cam in enumerate(cself.CameraChain.camList):
        print("cam{0}".format(camNr), file=stream)
        print("-----", file=stream)
        cam.camConfig.printDetails(stream)
        print("", file=stream)

        print("", file=stream)
    print("", file=stream)
    print("IMU configuration", file=stream)
    print("=================", file=stream)
    print("", file=stream)
    for (imuNr, imu) in enumerate(cself.ImuList):
        print("IMU{0}:\n".format(imuNr), "----------------------------", file=stream)
        imu.getImuConfig().printDetails(stream)
        print("", file=stream)

def showPointCloud(pointCloud, geometries=[], window_name=''):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=window_name)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    interval = 1. / len(pointCloud)
    for i, pc in enumerate(pointCloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.paint_uniform_color(colorsys.hsv_to_rgb(i * interval, 1, 1))
        visualizer.add_geometry(pcd)
    visualizer.get_render_option().point_size = 2.
    visualizer.run()
