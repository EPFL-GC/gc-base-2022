import igl
import pyvista as pv
import numpy as np

import fabrication_helper, tracer_tool
from tracer_tool import AsymptoticTracer
from fabrication_helper import FabricationHelper
import sys

filename = sys.argv[1] # "../data/Model01"

if filename.endswith(".obj"):
    filename = filename[:-4]

# Fabrication parameters in cm
strips_scale = 3
strips_width = 2
strips_spacing = 0.2
strips_thickness = 0.3
board_width = 60
board_length = 100
# Creation parameters
sampling_distance = 1.0
num_neighbors = 4
iter_sampling = False
# Visualisation
strips_actor = []
points_actor = []
labels_actor = []
samples_actor = {}
pathA_actor = {}
pathA_indexes = {}
pathB_actor = {}
pathB_indexes = {}
intersections = np.empty((0,3), float)
visual_scaling = 0.2

# Initialise tracer
mesh = pv.read(filename + ".obj")
v,f = igl.read_triangle_mesh(filename + ".obj")
tracer = AsymptoticTracer(filename + ".obj")
helper = FabricationHelper(strips_width, strips_thickness, strips_spacing, strips_scale)

plot = pv.Plotter(notebook=0)

def add_pathA(pid):
    points, samples = tracer.generate_asymptotic_path(pid, True, num_neighbors, sampling_distance) 

    if len(points)> 1:
        pathA_actor[pid] = plot.add_mesh(pv.Spline(points, 400), color='white', line_width=10, pickable=False)
        pathA_indexes[pid] = tracer.num_pathsA()-1
        tracer.samples_indexes[0].append(pid)
    return samples

def add_pathB(pid):
    points, samples = tracer.generate_asymptotic_path(pid, False, num_neighbors, sampling_distance) 
    if len(points)> 1:
        pathB_actor[pid] = plot.add_mesh(pv.Spline(points, 400), color='yellow', line_width=10, pickable=False)
        pathB_indexes[pid] = tracer.num_pathsB()-1
        tracer.samples_indexes[1].append(pid) 
    return samples

def remove_pathA(pid):
    plot.remove_actor(pathA_actor[pid])
    tracer.delete_path(pathA_indexes[pid], True)
    del pathA_actor[pid]
    #Update indexes
    update_indexes(pid, True)

def remove_pathB(pid):
    plot.remove_actor(pathB_actor[pid])
    tracer.delete_path(pathB_indexes[pid], False)
    del pathB_actor[pid]
    #Update indexes
    update_indexes(pid, False)

def add_or_delete_sample_point(pid):
    orig = v[pid]

    if pid not in pathB_actor.keys() and pid not in pathA_actor.keys():
        if pid in samples_actor.keys():
            plot.remove_actor(samples_actor[pid])
            del samples_actor[pid]
            clean_intersections()
    else:
        if pid not in samples_actor.keys():
            color = 'blue'
            if tracer.flagA and not tracer.flagB:
                color = 'white'
            elif tracer.flagB and not tracer.flagA:
                color = 'yellow'

            samples_actor[pid] = plot.add_points(np.array(orig), color=color, render_points_as_spheres=True, point_size=20.0, pickable=False)
            clean_intersections()
        else:
            plot.remove_actor(samples_actor[pid])
            color = 'blue'
            if pid not in pathB_actor.keys() and pid in pathA_actor.keys():
                color = 'white'
            elif pid in pathB_actor.keys() and pid not in pathA_actor.keys():
                color = 'yellow'
            samples_actor[pid] = plot.add_points(np.array(orig), color=color, render_points_as_spheres=True, point_size=20.0, pickable=False)
            clean_intersections()

def update_indexes(path_index, first_principal_direction):
    if first_principal_direction:
        if path_index in pathA_indexes.keys():
            del pathA_indexes[path_index]
            idx = 0
            for key in pathA_indexes:
                pathA_indexes[key] = idx
                idx+=1
    else:
        if path_index in pathB_indexes.keys():
            del pathB_indexes[path_index]
            idx = 0
            for key in pathB_indexes:
                pathB_indexes[key] = idx
                idx+=1

def callback_first_family(value):
    tracer.flagA = value

def callback_second_family(value):
    tracer.flagB = value

def clean_intersections():
    if len(points_actor)>0:
        callback_remove_labels()
        plot.remove_actor(points_actor)
        points_actor.clear()
        labels_actor.clear()
        tracer.flag_intersections = False

def callback_intersection():
    clean_intersections() 
    global intersections
    intersections = np.empty((0,3), float)
    intersections = np.append(intersections, tracer.generate_intersection_network(), axis=0)
    if len(tracer.intersection_points)>0:
        points_actor.append(plot.add_points(tracer.intersection_points, color='red',point_size=13.0, pickable=False)) 

def callback_flatten():
    if not tracer.flag_intersections:
        callback_intersection()
    
    helper.generate_flatten_network(tracer)
    
    strips_num = helper.strips_numA if helper.strips_numA > helper.strips_numB else helper.strips_numB
    plot.remove_actor(strips_actor)
    strips_actor.clear()
    if strips_num:
        for i in range(strips_num):
            if i<helper.strips_numA:
                points = helper.paths_flatten[0][i] * visual_scaling
                strips_actor.append(plot.add_lines(lines_from_points(points), color='white', width=3))
            if i<helper.strips_numB:
                points = helper.paths_flatten[1][i] * visual_scaling
                strips_actor.append(plot.add_lines(lines_from_points(points), color='yellow', width=3))
        # Board boundary
        points = np.array([[0.,0.,0.],[board_length*visual_scaling,0.,0.],[board_length*visual_scaling, board_width*visual_scaling,0.],[0., board_width*visual_scaling,0],[0.,0.,0.]])
        strips_actor.append(plot.add_lines(lines_from_points(points), color='red', width=3))
    
def callback_save_indexes():
    file = open(filename + "_indexes.txt", 'w')
    for i in range(len(tracer.samples_indexes)):
        for idx in tracer.samples_indexes[i]:
            if i==0:  
                file.write("A"+str(idx))
            elif i==1:
                file.write("B"+str(idx))
            file.write('\n')
    file.close()

def callback_load_indexes():
    indexes = np.loadtxt(filename + "_indexes.txt", dtype=str)
    old_flagA = tracer.flagA
    old_flagB = tracer.flagB

    for data in indexes:
        pid = int(data[1:])

        tracer.flagA = True if data[0] == "A" else False
        tracer.flagB = True if data[0] == "B" else False

        callback_picking(mesh,pid)

    tracer.flagA = old_flagA
    tracer.flagB = old_flagB

def lines_from_points(points):
    cells = np.empty((len(points)-1, 2), dtype=np.int_)
    cells[:,0] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:,1] = np.arange(1, len(points), dtype=np.int_)
    cells = cells.flatten()
    return np.array([points[i] for i in cells])

def callback_picking(mesh, pid, iterative_sampling=None):

    if(iterative_sampling==None):
        iterative_sampling = iter_sampling

    old_flagA = tracer.flagA
    old_flagB = tracer.flagB

    # Generate first family of asymptotic curves
    if tracer.flagA:
        if pid in pathA_actor.keys():
            remove_pathA(pid)
        else:
            samples = add_pathA(pid)

            if iterative_sampling:
                for pt in samples:
                    idx = tracer.mesh.kd_tree.query(pt)[1]
                    tracer.flagA = False
                    tracer.flagB = True
                    callback_picking(mesh, idx, iterative_sampling=False)

    tracer.flagA = old_flagA
    tracer.flagB = old_flagB

    # Generate second family of asymptotic curves
    if tracer.flagB:
        if pid in pathB_actor.keys():
            remove_pathB(pid)
        else:
            samples = add_pathB(pid)

            if iterative_sampling:
                for pt in samples:
                    idx = tracer.mesh.kd_tree.query(pt)[1]
                    tracer.flagA = True
                    tracer.flagB = False
                    callback_picking(mesh, idx, iterative_sampling=False)

    tracer.flagA = old_flagA
    tracer.flagB = old_flagB
    
    add_or_delete_sample_point(pid)
    
def callback_save_svg():
    cutting_color = "red"
    engraving_color = "black"
    font_size = 0.4
    if helper.flag_flatten==False:
        helper.generate_flatten_network()
    helper.generate_svg_file(filename + "_cutting.svg", font_size, cutting_color, engraving_color, board_length, board_width)

def callback_width(value):
    helper.strips_width = value
    callback_flatten()

def callback_board_width(value):
    global board_width
    board_width = value
    callback_flatten()

def callback_thickness(value):
    helper.strips_thickness = value
    callback_flatten()

def callback_length(value):
    helper.scale_length = value
    callback_flatten()

def callback_spacing(value):
    helper.strips_spacing = value
    callback_flatten()

def callback_board_length(value):
    global board_length
    board_length = value
    callback_flatten()

def callback_remove_labels():
    plot.remove_actor(labels_actor)
    labels_actor.clear()

def callback_add_all_labels():
    callback_add_labels(True, True)

def callback_add_labelsA():
    callback_add_labels(True, False)

def callback_add_labelsB():
    callback_add_labels(False, True)
    
def callback_add_labels(labelsA, labelsB):
    callback_remove_labels()

    if len(tracer.paths_indexes[0])>0 and labelsA:
        labels = np.core.defchararray.add('A', np.arange(len(tracer.paths_indexes[0])).astype(str))
        indexes = [idx[:1][0] for idx in tracer.paths_indexes[0]]
        labels_actor.append(plot.add_point_labels(tracer.paths[0][indexes], labels, font_size=22, always_visible=True, show_points=False))

        indexes = np.unique(np.array([item for sublist in tracer.intersections[0] for item in sublist[:,2]], int).flatten())
        labels = np.core.defchararray.add('c', indexes.astype(str))
        labels_actor.append(plot.add_point_labels(tracer.intersection_points[indexes], labels, bold=False, font_size=18, always_visible=True, show_points=False))

    if len(tracer.paths_indexes[1])>0 and labelsB:
        labels = np.core.defchararray.add('B', np.arange(len(tracer.paths_indexes[1])).astype(str))
        indexes = [idx[:1][0] for idx in tracer.paths_indexes[1]]
        labels_actor.append(plot.add_point_labels(tracer.paths[1][indexes], labels, font_size=22, always_visible=True, show_points=False))

        indexes = np.unique(np.array([item for sublist in tracer.intersections[1] for item in sublist[:,2]], int).flatten())
        labels = np.core.defchararray.add('c', indexes.astype(str))
        labels_actor.append(plot.add_point_labels(tracer.intersection_points[indexes], labels, bold=False, font_size=18, always_visible=True, show_points=False))

def callback_save_network():
    file = open(filename + "_rhino.txt", 'w')
    for i in range(2):
        label = "A"
        if i==1:
            label ="B"

        # positions
        for j in range(len(tracer.paths_indexes[i])):
            path = tracer.paths_indexes[i][j]
            for idx in path:
                pt = tracer.paths[i][idx]
                file.write(label + str(j)+ "_" +str(pt[0]) + "," + str(pt[1]) + "," +str(pt[2]) + "\n")

    # Intersections    
    for i in range( len(tracer.intersection_points) ):
        pt = tracer.intersection_points[i]
        file.write("C" + str(i) + "_" +str(pt[0]) + "," + str(pt[1]) + "," +str(pt[2]) + "\n")
    file.close()

def callback_sampling_distance(value):
    global sampling_distance
    sampling_distance = value

def callback_iterative_sampling(value):
    global iter_sampling
    iter_sampling = value

plot.add_mesh(mesh, show_edges=True)
plot.add_axes()
msg = "Press <K> for saving indexes, <L> for loading indexes or <O> to save the curve network model.\n"
msg += "Press <I> for computing intersections, <J> for generating the flatten strips and <H> for saving the laser-cutting file.\n"
msg += "Press <M> for hiding labels, <N> for showing all labels, <U> for showing A labels or <Y> for showing B labels.\n"
plot.add_text(msg, position='lower_right', font_size=12, color=None, font=None, shadow=False, name=None, viewport=False)
plot.add_checkbox_button_widget(callback_first_family, value=tracer.flagA, position=(10, 200.0), size=40, border_size=1, color_on='white', color_off='grey', background_color='red')
plot.add_checkbox_button_widget(callback_second_family, value=tracer.flagB, position=(10, 300.0), size=40, border_size=1, color_on='yellow', color_off='grey', background_color='red')
plot.add_checkbox_button_widget(callback_iterative_sampling, value=iter_sampling, position=(10, 400.0), size=40, border_size=1, color_on='green', color_off='grey', background_color='red')
plot.add_text("First Family", position=(80.0, 200.0), font_size=12, color=None, font=None, shadow=False, name=None, viewport=False)
plot.add_text("Second Family", position=(80, 300.0), font_size=12, color=None, font=None, shadow=False, name=None, viewport=False)
plot.add_text("Iterative sampling", position=(80, 400.0), font_size=12, color=None, font=None, shadow=False, name=None, viewport=False)
plot.add_key_event('i', callback_intersection)
plot.add_key_event('j', callback_flatten)
plot.add_key_event('k', callback_save_indexes)
plot.add_key_event('l', callback_load_indexes)
plot.add_key_event('h', callback_save_svg)
plot.add_key_event('m', callback_remove_labels)
plot.add_key_event('n', callback_add_all_labels)
plot.add_key_event('u', callback_add_labelsA)
plot.add_key_event('y', callback_add_labelsB)
plot.add_key_event('o', callback_save_network)
plot.enable_point_picking(callback=callback_picking, show_message=True, color='pink', point_size=10, use_mesh=True, show_point=True)
plot.add_slider_widget(callback_width, [0.1, 5.0], value=strips_width, title="Strip Width (cm)", pointa=(.83, .15), pointb=(.98, .15), title_height=0.02, fmt="%0.1f", style='modern')
plot.add_slider_widget(callback_thickness, [0.1, 1], value=strips_thickness, title="Strip Thickness (cm)", pointa=(.67, .15), pointb=(.82, .15), title_height=0.02, fmt="%0.2f", style='modern')
plot.add_slider_widget(callback_length, [1, 10], value=strips_scale, title="Scale Strip Length", pointa=(.51, .15), pointb=(.66, .15), title_height=0.02, fmt="%0.1f", style='modern')
plot.add_slider_widget(callback_spacing, [0., 0.5], value=strips_spacing, title="Strip spacing", pointa=(.51, .88), pointb=(.66, .88), title_height=0.02, fmt="%0.1f", style='modern')
plot.add_slider_widget(callback_board_width, [10, 100], value=board_width, title="Board width (cm)", pointa=(.67, .88), pointb=(.82, .88), title_height=0.02, fmt="%0.0f", style='modern')
plot.add_slider_widget(callback_board_length, [10, 100], value=board_length, title="Board length (cm)", pointa=(.83, .88), pointb=(.98, .88), title_height=0.02, fmt="%0.0f", style='modern')
plot.add_slider_widget(callback_sampling_distance, [0.1, 2.0], value=sampling_distance, title="Sampling distance", pointa=(.005, .88), pointb=(.16, .88), title_height=0.02, fmt="%0.1f", style='modern')
plot.show("Asymptotic GridShell")
