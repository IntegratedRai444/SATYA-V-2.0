"""
SatyaAI - 3D Animations Module
This module provides 3D visualizations for the deepfake detection system
"""
import os
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D
import time
import random

class AnimationGenerator:
    """Base class for all animations"""
    
    def __init__(self, size=(800, 600), dpi=100):
        """Initialize with figure size and DPI"""
        self.size = size
        self.dpi = dpi
        self.theme_colors = {
            'primary': '#0ff5fc',
            'secondary': '#0070ff',
            'dark': '#0a1420',
            'darker': '#050a14',
            'light_text': '#e9f8ff',
            'success': '#22ff22',
            'warning': '#ff9500',
            'danger': '#ff3b30',
        }
        # Set global plot style
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = self.theme_colors['darker']
        plt.rcParams['figure.facecolor'] = self.theme_colors['darker']
        plt.rcParams['text.color'] = self.theme_colors['light_text']
        plt.rcParams['axes.labelcolor'] = self.theme_colors['light_text']
        plt.rcParams['xtick.color'] = self.theme_colors['light_text']
        plt.rcParams['ytick.color'] = self.theme_colors['light_text']
    
    def save_frame(self, fig, output_path=None):
        """Save a single frame to a file or return as base64"""
        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            return output_path
        else:
            # Save to buffer and convert to base64
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
    
    def create_animation(self, frames=30, fps=15, output_path=None):
        """Create animation and save to file or return as base64"""
        raise NotImplementedError("Subclasses must implement this method")


class NeuralNetworkAnimation(AnimationGenerator):
    """Create 3D neural network animation for deepfake detection visualization"""
    
    def __init__(self, size=(800, 600), dpi=100):
        super().__init__(size, dpi)
        
        # Neural network parameters
        self.layers = [8, 16, 32, 64, 32, 16, 8, 4, 2]  # Nodes per layer
        self.layer_spacing = 0.5
        
        # Generate initial node positions
        self.nodes = []
        for i, layer_size in enumerate(self.layers):
            layer = []
            for j in range(layer_size):
                # Position nodes in a circle
                angle = 2 * np.pi * j / layer_size
                radius = 0.3 + 0.05 * layer_size
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = i * self.layer_spacing
                
                # Add node with random activation
                layer.append({
                    'pos': (x, y, z),
                    'activation': random.random()
                })
            self.nodes.append(layer)
        
        # Generate connections between layers
        self.connections = []
        for i in range(len(self.layers) - 1):
            layer_connections = []
            # Connect each node in current layer to every node in next layer
            for node1 in self.nodes[i]:
                for node2 in self.nodes[i+1]:
                    # Random weight for connection
                    weight = random.random()
                    layer_connections.append({
                        'start': node1['pos'],
                        'end': node2['pos'],
                        'weight': weight
                    })
            self.connections.append(layer_connections)
    
    def create_static_frame(self, authenticity="AUTHENTIC MEDIA", confidence=85):
        """Create a static neural network visualization"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        ax.set_facecolor(self.theme_colors['darker'])
        
        # Determine color based on authenticity
        main_color = self.theme_colors['success'] if authenticity == "AUTHENTIC MEDIA" else self.theme_colors['danger']
        
        # Plot connections
        for i, layer_connections in enumerate(self.connections):
            for conn in layer_connections:
                xs = [conn['start'][0], conn['end'][0]]
                ys = [conn['start'][1], conn['end'][1]]
                zs = [conn['start'][2], conn['end'][2]]
                
                # Color based on weight and confidence
                weight_color = conn['weight'] * (confidence / 100)
                ax.plot(xs, ys, zs, alpha=weight_color*0.5, 
                        color=main_color, linewidth=weight_color*2)
        
        # Plot nodes
        for i, layer in enumerate(self.nodes):
            for node in layer:
                x, y, z = node['pos']
                
                # Size and color based on activation and confidence
                size = 50 + 150 * node['activation'] * (confidence / 100)
                
                # Determine color for node
                if i == 0:  # Input layer
                    color = self.theme_colors['primary']
                elif i == len(self.nodes) - 1:  # Output layer
                    color = main_color
                else:  # Hidden layers
                    color = self.theme_colors['secondary']
                
                ax.scatter(x, y, z, s=size, c=color, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Add title and labels
        title_y_pos = 1.05
        title = fig.suptitle(f"Neural Network Analysis: {authenticity}", 
                           color=main_color, fontsize=16, y=title_y_pos, fontweight='bold')
        
        confidence_text = ax.text2D(0.5, 0.95, f"Confidence: {confidence:.1f}%", 
                                   transform=ax.transAxes, fontsize=12, ha='center',
                                   color=self.theme_colors['light_text'])
        
        # Add SatyaAI logo/watermark
        fig.text(0.05, 0.02, "SatyaAI", fontsize=14, color=self.theme_colors['primary'], 
                 fontstyle='italic', fontweight='bold')
        
        # Set the view angle
        ax.view_init(elev=20, azim=30)
        
        # Remove axis labels and ticks
        ax.set_axis_off()
        
        # Set plot limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, (len(self.layers) - 1) * self.layer_spacing)
        
        return fig
    
    def create_animation(self, frames=30, fps=15, authenticity="AUTHENTIC MEDIA", confidence=85, output_path=None):
        """Create a neural network animation"""
        # Function to update the plot for each frame
        def update(frame):
            # Create a new figure for each frame
            fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Set background color
            ax.set_facecolor(self.theme_colors['darker'])
            
            # Determine color based on authenticity
            main_color = self.theme_colors['success'] if authenticity == "AUTHENTIC MEDIA" else self.theme_colors['danger']
            
            # Update activation values for animation
            for layer in self.nodes:
                for node in layer:
                    # Adjust activation with some random fluctuation
                    node['activation'] = min(1.0, max(0.1, node['activation'] + 
                                                    random.uniform(-0.1, 0.1)))
            
            # Update connection weights
            for layer_connections in self.connections:
                for conn in layer_connections:
                    # Adjust weight with some random fluctuation
                    conn['weight'] = min(1.0, max(0.1, conn['weight'] + 
                                                  random.uniform(-0.05, 0.05)))
            
            # Rotate the view angle
            ax.view_init(elev=20, azim=30 + frame * (360/frames))
            
            # Plot connections with animation effects
            for i, layer_connections in enumerate(self.connections):
                for conn in layer_connections:
                    xs = [conn['start'][0], conn['end'][0]]
                    ys = [conn['start'][1], conn['end'][1]]
                    zs = [conn['start'][2], conn['end'][2]]
                    
                    # Pulse effect on connection opacity
                    pulse = 0.5 + 0.5 * np.sin(frame * 0.2 + i * 0.5)
                    
                    # Color based on weight, confidence and pulse
                    weight_color = conn['weight'] * (confidence / 100) * pulse
                    ax.plot(xs, ys, zs, alpha=weight_color*0.5, 
                            color=main_color, linewidth=weight_color*2)
            
            # Plot nodes with animation effects
            for i, layer in enumerate(self.nodes):
                for j, node in enumerate(layer):
                    x, y, z = node['pos']
                    
                    # Pulse effect on node size
                    pulse = 1.0 + 0.3 * np.sin(frame * 0.2 + i * 0.5 + j * 0.2)
                    
                    # Size and color based on activation, confidence and pulse
                    size = (50 + 150 * node['activation'] * (confidence / 100)) * pulse
                    
                    # Determine color for node
                    if i == 0:  # Input layer
                        color = self.theme_colors['primary']
                    elif i == len(self.nodes) - 1:  # Output layer
                        color = main_color
                    else:  # Hidden layers
                        color = self.theme_colors['secondary']
                    
                    ax.scatter(x, y, z, s=size, c=color, alpha=0.8, edgecolors='white', linewidth=0.5)
            
            # Add title and labels
            title_y_pos = 1.05
            title = fig.suptitle(f"Neural Network Analysis: {authenticity}", 
                               color=main_color, fontsize=16, y=title_y_pos, fontweight='bold')
            
            confidence_text = ax.text2D(0.5, 0.95, f"Confidence: {confidence:.1f}%", 
                                       transform=ax.transAxes, fontsize=12, ha='center',
                                       color=self.theme_colors['light_text'])
            
            # Add SatyaAI logo/watermark
            fig.text(0.05, 0.02, "SatyaAI", fontsize=14, color=self.theme_colors['primary'], 
                     fontstyle='italic', fontweight='bold')
                     
            # Add processing text with frame counter
            process_text = f"Processing frame {frame+1}/{frames}"
            fig.text(0.95, 0.02, process_text, fontsize=10, color=self.theme_colors['light_text'],
                     ha='right')
            
            # Remove axis labels and ticks
            ax.set_axis_off()
            
            # Set plot limits
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(0, (len(self.layers) - 1) * self.layer_spacing)
            
            # Save the frame
            frame_data = self.save_frame(fig, 
                                         output_path=f"{output_path}_frame_{frame}.png" if output_path else None)
            
            # Close the figure to avoid memory leak
            plt.close(fig)
            
            return frame_data
        
        # Generate and collect all frames
        frame_data = []
        for i in range(frames):
            frame_data.append(update(i))
        
        return frame_data


class BlockchainAnimation(AnimationGenerator):
    """Create 3D blockchain animation for SatyaChain™"""
    
    def __init__(self, size=(800, 600), dpi=100):
        super().__init__(size, dpi)
        
        # Generate blockchain structure
        self.num_blocks = 10
        self.blocks = []
        
        # Create blockchain blocks with random data
        for i in range(self.num_blocks):
            self.blocks.append({
                'index': i,
                'pos': (0, 0, i),
                'hash': ''.join(random.choices('0123456789abcdef', k=16)),
                'prev_hash': ''.join(random.choices('0123456789abcdef', k=16)) if i > 0 else '0' * 16,
                'rotation': random.uniform(0, 360)
            })
    
    def create_static_frame(self, is_verified=True, verification_progress=100):
        """Create a static blockchain visualization"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        ax.set_facecolor(self.theme_colors['darker'])
        
        # Determine main color based on verification status
        main_color = self.theme_colors['success'] if is_verified else self.theme_colors['danger']
        
        # Plot blockchain blocks
        for i, block in enumerate(self.blocks):
            # Only show blocks up to verification progress
            if (i / self.num_blocks) * 100 <= verification_progress:
                # Block dimensions
                width, height, depth = 0.8, 0.4, 0.3
                
                # Block position
                x, y, z = 0, 0, i * 0.6
                
                # Create a cuboid to represent the block
                xx, yy = np.meshgrid(
                    [x - width/2, x + width/2],
                    [y - height/2, y + height/2]
                )
                
                # Draw block faces
                # Bottom face
                ax.plot_surface(xx, yy, np.ones_like(xx) * (z - depth/2),
                                color=main_color, alpha=0.7, edgecolor='white', shade=True)
                
                # Top face
                ax.plot_surface(xx, yy, np.ones_like(xx) * (z + depth/2),
                                color=main_color, alpha=0.7, edgecolor='white', shade=True)
                
                # Side faces
                for dx, dy in [(-width/2, 0), (width/2, 0), (0, -height/2), (0, height/2)]:
                    if dx != 0:  # X-facing side
                        x_face = np.ones((2, 2)) * (x + dx)
                        y_face = np.array([[y - height/2, y + height/2], [y - height/2, y + height/2]])
                        z_face = np.array([[z - depth/2, z - depth/2], [z + depth/2, z + depth/2]])
                    else:  # Y-facing side
                        x_face = np.array([[x - width/2, x + width/2], [x - width/2, x + width/2]])
                        y_face = np.ones((2, 2)) * (y + dy)
                        z_face = np.array([[z - depth/2, z - depth/2], [z + depth/2, z + depth/2]])
                    
                    ax.plot_surface(x_face, y_face, z_face,
                                    color=main_color, alpha=0.7, edgecolor='white', shade=True)
                
                # Draw hash text
                ax.text(x, y, z, f"#{block['index']} {block['hash'][:4]}...",
                        color=self.theme_colors['light_text'], fontsize=8,
                        horizontalalignment='center', verticalalignment='center')
                
                # Draw connections between blocks
                if i > 0:
                    ax.plot([0, 0], [0, 0], [z - 0.6, z - 0.3],
                            color=main_color, linewidth=2, linestyle='-')
        
        # Add title based on verification status
        title_text = "BLOCKCHAIN VERIFIED" if is_verified else "VERIFICATION FAILED"
        title = fig.suptitle(title_text, color=main_color, fontsize=16, y=1.05, fontweight='bold')
        
        # Add verification progress
        progress_text = ax.text2D(0.5, 0.95, f"Verification Progress: {verification_progress:.1f}%", 
                                 transform=ax.transAxes, fontsize=12, ha='center',
                                 color=self.theme_colors['light_text'])
        
        # Add SatyaChain watermark
        fig.text(0.05, 0.02, "SatyaChain™", fontsize=14, color=self.theme_colors['primary'], 
                 fontstyle='italic', fontweight='bold')
        
        # Set the view angle
        ax.view_init(elev=30, azim=45)
        
        # Remove axis labels and ticks
        ax.set_axis_off()
        
        # Set plot limits
        margin = 1
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_zlim(-1, self.num_blocks * 0.6)
        
        return fig
    
    def create_animation(self, frames=30, fps=15, is_verified=True, output_path=None):
        """Create a blockchain animation"""
        # Function to update the plot for each frame
        def update(frame):
            # Create a new figure for each frame
            fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Set background color
            ax.set_facecolor(self.theme_colors['darker'])
            
            # Calculate verification progress for this frame
            progress = min(100, (frame + 1) / frames * 100 * 2)  # Complete by 50% of frames
            
            # Determine color based on verification status and current progress
            if is_verified:
                main_color = self.theme_colors['success'] if progress == 100 else self.theme_colors['primary']
            else:
                main_color = self.theme_colors['danger'] if progress == 100 else self.theme_colors['warning']
            
            # Plot blockchain blocks
            for i, block in enumerate(self.blocks):
                # Only show blocks up to verification progress
                if (i / self.num_blocks) * 100 <= progress:
                    # Block dimensions
                    width, height, depth = 0.8, 0.4, 0.3
                    
                    # Block position with slight movement
                    x = 0.05 * np.sin(frame * 0.1 + i * 0.5)
                    y = 0.05 * np.cos(frame * 0.1 + i * 0.5)
                    z = i * 0.6
                    
                    # Rotation for animation
                    angle = block['rotation'] + frame * 5
                    
                    # Create a cuboid to represent the block
                    xx, yy = np.meshgrid(
                        [x - width/2, x + width/2],
                        [y - height/2, y + height/2]
                    )
                    
                    # Draw block faces with animation effects
                    alpha = 0.7 + 0.3 * np.sin(frame * 0.1 + i * 0.5)
                    
                    # Bottom face
                    ax.plot_surface(xx, yy, np.ones_like(xx) * (z - depth/2),
                                    color=main_color, alpha=alpha, edgecolor='white', shade=True)
                    
                    # Top face
                    ax.plot_surface(xx, yy, np.ones_like(xx) * (z + depth/2),
                                    color=main_color, alpha=alpha, edgecolor='white', shade=True)
                    
                    # Side faces
                    for dx, dy in [(-width/2, 0), (width/2, 0), (0, -height/2), (0, height/2)]:
                        if dx != 0:  # X-facing side
                            x_face = np.ones((2, 2)) * (x + dx)
                            y_face = np.array([[y - height/2, y + height/2], [y - height/2, y + height/2]])
                            z_face = np.array([[z - depth/2, z - depth/2], [z + depth/2, z + depth/2]])
                        else:  # Y-facing side
                            x_face = np.array([[x - width/2, x + width/2], [x - width/2, x + width/2]])
                            y_face = np.ones((2, 2)) * (y + dy)
                            z_face = np.array([[z - depth/2, z - depth/2], [z + depth/2, z + depth/2]])
                        
                        ax.plot_surface(x_face, y_face, z_face,
                                        color=main_color, alpha=alpha, edgecolor='white', shade=True)
                    
                    # Draw hash text with blinking effect
                    blink_alpha = 0.5 + 0.5 * np.sin(frame * 0.5 + i)
                    ax.text(x, y, z, f"#{block['index']} {block['hash'][:4]}...",
                            color=self.theme_colors['light_text'], fontsize=8, alpha=blink_alpha,
                            horizontalalignment='center', verticalalignment='center')
                    
                    # Draw connections between blocks with pulsing effect
                    if i > 0:
                        pulse = 1 + 0.5 * np.sin(frame * 0.3 + i)
                        ax.plot([x_prev, x], [y_prev, y], [z_prev + depth/2, z - depth/2],
                                color=main_color, linewidth=1 * pulse, linestyle='-')
                    
                    # Store previous block coordinates for connection
                    x_prev, y_prev, z_prev = x, y, z
            
            # Add title based on verification status
            if progress < 100:
                title_text = f"BLOCKCHAIN VERIFICATION IN PROGRESS ({progress:.0f}%)"
                title_color = self.theme_colors['primary']
            else:
                title_text = "BLOCKCHAIN VERIFIED" if is_verified else "VERIFICATION FAILED"
                title_color = self.theme_colors['success'] if is_verified else self.theme_colors['danger']
            
            title = fig.suptitle(title_text, color=title_color, fontsize=16, y=1.05, fontweight='bold')
            
            # Add verification progress bar as a text visualization
            bar_length = 20
            completed = int(progress / 100 * bar_length)
            progress_bar = '[' + '■' * completed + '□' * (bar_length - completed) + ']'
            progress_text = ax.text2D(0.5, 0.95, f"{progress_bar} {progress:.1f}%", 
                                     transform=ax.transAxes, fontsize=10, ha='center',
                                     color=self.theme_colors['light_text'])
            
            # Add SatyaChain watermark
            fig.text(0.05, 0.02, "SatyaChain™", fontsize=14, color=self.theme_colors['primary'], 
                     fontstyle='italic', fontweight='bold')
                     
            # Add processing text with frame counter
            process_text = f"Processing frame {frame+1}/{frames}"
            fig.text(0.95, 0.02, process_text, fontsize=10, color=self.theme_colors['light_text'],
                     ha='right')
            
            # Set the view angle with rotation for animation
            elevation = 20 + 10 * np.sin(frame * 0.1)
            azimuth = 30 + frame * (180 / frames)
            ax.view_init(elev=elevation, azim=azimuth)
            
            # Remove axis labels and ticks
            ax.set_axis_off()
            
            # Set plot limits
            margin = 1
            ax.set_xlim(-margin, margin)
            ax.set_ylim(-margin, margin)
            ax.set_zlim(-1, self.num_blocks * 0.6)
            
            # Save the frame
            frame_data = self.save_frame(fig, 
                                         output_path=f"{output_path}_frame_{frame}.png" if output_path else None)
            
            # Close the figure to avoid memory leak
            plt.close(fig)
            
            return frame_data
        
        # Generate and collect all frames
        frame_data = []
        for i in range(frames):
            frame_data.append(update(i))
        
        return frame_data


class WaveformAnimation(AnimationGenerator):
    """Create audio waveform animation for multi-language lip-sync analysis"""
    
    def __init__(self, size=(800, 600), dpi=100, language="english"):
        super().__init__(size, dpi)
        
        # Language settings
        self.language = language
        self.language_settings = {
            'english': {
                'color': self.theme_colors['primary'],
                'display_name': 'English',
                'wave_freq': 1.0,
                'wave_complexity': 1.0
            },
            'hindi': {
                'color': '#FFA500',  # Orange
                'display_name': 'Hindi',
                'wave_freq': 1.2,
                'wave_complexity': 1.3
            },
            'mandarin': {
                'color': '#FF0000',  # Red
                'display_name': 'Mandarin',
                'wave_freq': 1.5,
                'wave_complexity': 1.6
            },
            'french': {
                'color': '#0000FF',  # Blue
                'display_name': 'French',
                'wave_freq': 0.8,
                'wave_complexity': 1.2
            },
            'spanish': {
                'color': '#FFFF00',  # Yellow
                'display_name': 'Spanish',
                'wave_freq': 1.1,
                'wave_complexity': 1.1
            }
        }
        
        # Default to English if language not found
        if language not in self.language_settings:
            self.language = 'english'
        
        # Generate waveform data
        self.waveform_length = 100
        self.waveform_data = self.generate_waveform()
        self.lip_sync_data = self.generate_lip_sync_data()
    
    def generate_waveform(self):
        """Generate synthetic waveform data based on language"""
        settings = self.language_settings[self.language]
        
        # Basic sine wave with language-specific frequency
        x = np.linspace(0, 10, self.waveform_length)
        base_wave = np.sin(x * settings['wave_freq'])
        
        # Add complexity based on language
        complex_wave = base_wave.copy()
        for i in range(1, int(5 * settings['wave_complexity'])):
            harmonic = np.sin(x * settings['wave_freq'] * (i + 1)) / (i + 1)
            complex_wave += harmonic * random.uniform(0.1, 0.3)
        
        # Normalize
        complex_wave = complex_wave / np.max(np.abs(complex_wave))
        
        return complex_wave
    
    def generate_lip_sync_data(self):
        """Generate synthetic lip sync data for the waveform"""
        settings = self.language_settings[self.language]
        
        # Generate lip movement data
        lip_data = []
        for i in range(self.waveform_length):
            # Lip positions (open/close)
            lip_position = 0.3 + 0.7 * abs(self.waveform_data[i])
            
            # Add mouth shape variations based on language
            shape_variation = random.uniform(0.8, 1.2) * settings['wave_complexity']
            
            lip_data.append({
                'position': lip_position,
                'shape': shape_variation
            })
        
        return lip_data
    
    def create_static_frame(self, sync_score=85):
        """Create a static waveform visualization"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        ax.set_facecolor(self.theme_colors['darker'])
        
        # Determine color based on sync score
        if sync_score >= 80:
            main_color = self.theme_colors['success']
        elif sync_score >= 60:
            main_color = self.theme_colors['warning']
        else:
            main_color = self.theme_colors['danger']
        
        # Language-specific color
        language_color = self.language_settings[self.language]['color']
        
        # Create a 3D waveform visualization
        x = np.arange(self.waveform_length)
        y = np.zeros_like(x)
        z = self.waveform_data
        
        # Line plot for waveform
        ax.plot(x, y, z, color=language_color, linewidth=2)
        
        # Add 3D markers for key points
        for i in range(0, self.waveform_length, 5):
            ax.scatter(x[i], y[i], z[i], 
                      color=language_color, s=30, alpha=0.7, edgecolors='white')
        
        # Add a reflection effect
        ax.plot(x, y - 0.5, -z * 0.3, color=language_color, linewidth=1, alpha=0.3)
        
        # Add lip sync visualization
        for i in range(0, self.waveform_length, 10):
            # Create a circle to represent mouth position
            lip_data = self.lip_sync_data[i]
            theta = np.linspace(0, 2 * np.pi, 20)
            
            # Ellipse coordinates to represent mouth shape
            a = lip_data['position']  # horizontal radius
            b = lip_data['position'] * lip_data['shape']  # vertical radius
            
            x_lip = x[i] + a * np.cos(theta)
            y_lip = -1 + np.zeros_like(theta)
            z_lip = z[i] + b * np.sin(theta)
            
            ax.plot(x_lip, y_lip, z_lip, color=main_color, alpha=0.7)
        
        # Add title and labels
        language_name = self.language_settings[self.language]['display_name']
        title = fig.suptitle(f"Lip Sync Analysis: {language_name}", 
                           color=language_color, fontsize=16, y=1.05, fontweight='bold')
        
        # Add sync score
        if sync_score >= 80:
            sync_text = "High Sync Probability"
        elif sync_score >= 60:
            sync_text = "Medium Sync Probability"
        else:
            sync_text = "Low Sync Probability"
            
        score_text = ax.text2D(0.5, 0.95, f"Sync Score: {sync_score}% - {sync_text}", 
                              transform=ax.transAxes, fontsize=12, ha='center',
                              color=main_color)
        
        # Add SatyaAI logo/watermark
        fig.text(0.05, 0.02, "SatyaAI", fontsize=14, color=self.theme_colors['primary'], 
                 fontstyle='italic', fontweight='bold')
        
        # Set the view angle
        ax.view_init(elev=20, azim=30)
        
        # Remove axis labels and ticks
        ax.set_axis_off()
        
        # Set plot limits
        ax.set_xlim(0, self.waveform_length)
        ax.set_ylim(-1.5, 0.5)
        ax.set_zlim(-1.5, 1.5)
        
        return fig
    
    def create_animation(self, frames=30, fps=15, sync_score=85, output_path=None):
        """Create a waveform animation"""
        # Function to update the plot for each frame
        def update(frame):
            # Create a new figure for each frame
            fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Set background color
            ax.set_facecolor(self.theme_colors['darker'])
            
            # Determine color based on sync score
            if sync_score >= 80:
                main_color = self.theme_colors['success']
            elif sync_score >= 60:
                main_color = self.theme_colors['warning']
            else:
                main_color = self.theme_colors['danger']
            
            # Language-specific color
            language_color = self.language_settings[self.language]['color']
            
            # Create a 3D waveform visualization with animation
            x = np.arange(self.waveform_length)
            y = np.zeros_like(x)
            
            # Animate the waveform with a ripple effect
            ripple = np.roll(self.waveform_data, frame * 2 % self.waveform_length)
            z = ripple + 0.2 * np.sin(x * 0.1 + frame * 0.2)
            
            # Line plot for waveform
            ax.plot(x, y, z, color=language_color, linewidth=2)
            
            # Add 3D markers for key points with pulsing effect
            for i in range(0, self.waveform_length, 5):
                pulse = 20 + 20 * np.sin(frame * 0.2 + i * 0.1)
                ax.scatter(x[i], y[i], z[i], 
                          color=language_color, s=pulse, alpha=0.7, edgecolors='white')
            
            # Add a reflection effect
            ax.plot(x, y - 0.5, -z * 0.3, color=language_color, linewidth=1, alpha=0.3)
            
            # Add lip sync visualization with animation
            for i in range(0, self.waveform_length, 10):
                # Create a circle to represent mouth position
                lip_data = self.lip_sync_data[i]
                theta = np.linspace(0, 2 * np.pi, 20)
                
                # Animate mouth shape
                phase = frame * 0.2 + i * 0.05
                a = lip_data['position'] * (0.8 + 0.2 * np.sin(phase))  # horizontal radius
                b = lip_data['position'] * lip_data['shape'] * (0.8 + 0.2 * np.sin(phase + np.pi/2))  # vertical radius
                
                x_lip = x[i] + a * np.cos(theta)
                y_lip = -1 + np.zeros_like(theta)
                z_lip = z[i] + b * np.sin(theta)
                
                ax.plot(x_lip, y_lip, z_lip, color=main_color, alpha=0.7)
            
            # Add moving analysis lines at current frame position
            scan_pos = frame % self.waveform_length
            if scan_pos < self.waveform_length - 1:
                # Vertical scan line
                ax.plot([scan_pos, scan_pos], [0, 0], [-1.5, 1.5], 
                        color='white', linewidth=1, alpha=0.5, linestyle='--')
                
                # Horizontal scan line at current height
                ax.plot([0, self.waveform_length], [0, 0], [z[scan_pos], z[scan_pos]], 
                        color='white', linewidth=1, alpha=0.3, linestyle=':')
            
            # Add title and labels
            language_name = self.language_settings[self.language]['display_name']
            title = fig.suptitle(f"Lip Sync Analysis: {language_name}", 
                               color=language_color, fontsize=16, y=1.05, fontweight='bold')
            
            # Add sync score with animation
            animated_score = sync_score * (0.9 + 0.1 * np.sin(frame * 0.2))
            
            if animated_score >= 80:
                sync_text = "High Sync Probability"
            elif animated_score >= 60:
                sync_text = "Medium Sync Probability"
            else:
                sync_text = "Low Sync Probability"
                
            score_text = ax.text2D(0.5, 0.95, f"Sync Score: {animated_score:.1f}% - {sync_text}", 
                                  transform=ax.transAxes, fontsize=12, ha='center',
                                  color=main_color)
            
            # Add SatyaAI logo/watermark
            fig.text(0.05, 0.02, "SatyaAI", fontsize=14, color=self.theme_colors['primary'], 
                     fontstyle='italic', fontweight='bold')
                     
            # Add processing text with frame counter
            process_text = f"Processing frame {frame+1}/{frames}"
            fig.text(0.95, 0.02, process_text, fontsize=10, color=self.theme_colors['light_text'],
                     ha='right')
            
            # Set the view angle with rotation for animation
            ax.view_init(elev=20, azim=30 + frame * (360/frames))
            
            # Remove axis labels and ticks
            ax.set_axis_off()
            
            # Set plot limits
            ax.set_xlim(0, self.waveform_length)
            ax.set_ylim(-1.5, 0.5)
            ax.set_zlim(-1.5, 1.5)
            
            # Save the frame
            frame_data = self.save_frame(fig, 
                                         output_path=f"{output_path}_frame_{frame}.png" if output_path else None)
            
            # Close the figure to avoid memory leak
            plt.close(fig)
            
            return frame_data
        
        # Generate and collect all frames
        frame_data = []
        for i in range(frames):
            frame_data.append(update(i))
        
        return frame_data