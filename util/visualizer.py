import numpy as np
import os
import sys
import ntpath
import time
from subprocess import Popen, PIPE


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt=None):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML files
        Step 4: create a logging file to store training losses
        """
        self.opt = opt if opt is not None else type('obj', (object,), {})()
        self.display_id = getattr(self.opt, 'display_id', 1)
        self.use_html = getattr(self.opt, 'isTrain', True) and not getattr(self.opt, 'no_html', False)
        self.win_size = getattr(self.opt, 'display_winsize', 256)
        self.name = getattr(self.opt, 'name', 'experiment_name')
        self.port = getattr(self.opt, 'display_port', 8097)
        self.saved = False
        
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = getattr(self.opt, 'display_ncols', 4)
            self.vis = visdom.Visdom(server=getattr(self.opt, 'display_server', "http://localhost"), port=self.port, env=getattr(self.opt, 'display_env', 'main'))
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(getattr(self.opt, 'checkpoints_dir', './checkpoints'), self.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            os.makedirs(self.web_dir, exist_ok=True)
            os.makedirs(self.img_dir, exist_ok=True)
        # create a logging file to store training losses
        self.log_name = os.path.join(getattr(self.opt, 'checkpoints_dir', './checkpoints'), self.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    # Convert tensor to numpy array
                    image_numpy = image.detach().cpu().numpy()
                    if image_numpy.ndim == 4:
                        image_numpy = image_numpy[0]
                    image_numpy = np.transpose(image_numpy, (1, 2, 0))
                    image_numpy = (image_numpy + 1) / 2.0 * 255.0
                    image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
                    
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        # Convert tensor to numpy array
                        image_numpy = image.detach().cpu().numpy()
                        if image_numpy.ndim == 4:
                            image_numpy = image_numpy[0]
                        image_numpy = np.transpose(image_numpy, (1, 2, 0))
                        image_numpy = (image_numpy + 1) / 2.0 * 255.0
                        image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
                        
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except Exception:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                # Convert tensor to numpy array
                image_numpy = image.detach().cpu().numpy()
                if image_numpy.ndim == 4:
                    image_numpy = image_numpy[0]
                image_numpy = np.transpose(image_numpy, (1, 2, 0))
                image_numpy = (image_numpy + 1) / 2.0 * 255.0
                image_numpy = np.clip(image_numpy, 0, 255).astype(np.uint8)
                
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                from PIL import Image
                Image.fromarray(image_numpy).save(img_path)

            # Simple HTML file creation (without dominate dependency)
            html_file = os.path.join(self.web_dir, 'index.html')
            with open(html_file, 'w') as f:
                f.write(f'<html><head><title>Experiment: {self.name}</title></head><body>\n')
                f.write(f'<h1>Experiment: {self.name}</h1>\n')
                f.write(f'<h2>Epoch {epoch}</h2>\n')
                for label, image in visuals.items():
                    img_filename = f'epoch{epoch:03d}_{label}.png'
                    img_path = os.path.join('images', img_filename)
                    f.write(f'<h3>{label}</h3>\n')
                    f.write(f'<img src="{img_path}" width="{self.win_size}"><br>\n')
                f.write('</body></html>\n')

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except Exception:
            self.create_visdom_connections()

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


# Simplified version for basic usage without visdom dependencies
class SimpleVisualizer:
    """Simplified visualizer that only handles logging without visdom"""
    
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_name = os.path.join(log_dir, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def print_current_losses(self, epoch, iters, losses, t_comp=0, t_data=0):
        """print current losses on console; also save the losses to the disk"""
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)