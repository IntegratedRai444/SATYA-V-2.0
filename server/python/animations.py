"""
SatyaAI - 3D Animations Module
This module provides 3D visualizations for the deepfake detection system
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import os
import time
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)

class AnimationGenerator:
    """Base class for all animations"""
    
    def __init__(self, size=(800, 600), dpi=100):
        self.size = size
        self.dpi = dpi
        
    def save_frame(self, fig, output_path=None):
        """Save a single frame to a file or return as base64"""
        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            return output_path
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('ascii')
            return img_str
    
    def create_animation(self, frames=30, fps=15, output_path=None):
        """Create animation and save to file or return as base64"""
        raise NotImplementedError("Subclasses must implement this method")


class NeuralNetworkAnimation(AnimationGenerator):
    """Create 3D neural network animation for deepfake detection visualization"""
    
    def __init__(self, size=(800, 600), dpi=100):
        super().__init__(size, dpi)
    
    def create_static_frame(self, authenticity="AUTHENTIC MEDIA", confidence=85):
        """Create a static neural network visualization"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        fig.patch.set_facecolor('#0A1128')
        ax.set_facecolor('#0A1128')
        
        # Remove axes and grid
        ax.set_axis_off()
        
        # Create network layers
        layer_sizes = [8, 16, 32, 16, 8, 4, 1]
        layer_positions = np.linspace(-3, 3, len(layer_sizes))
        
        # Generate node positions
        nodes = []
        for i, size in enumerate(layer_sizes):
            z = layer_positions[i]
            for j in range(size):
                angle = j * 2 * np.pi / size
                r = 1.5 * np.sqrt(size) / 4  # Radius based on layer size
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                nodes.append((x, y, z))
        
        nodes = np.array(nodes)
        
        # Generate connections between adjacent layers
        connections = []
        node_index = 0
        for i in range(len(layer_sizes) - 1):
            layer1_size = layer_sizes[i]
            layer2_size = layer_sizes[i + 1]
            
            for j in range(layer1_size):
                idx1 = node_index + j
                for k in range(layer2_size):
                    idx2 = node_index + layer1_size + k
                    connections.append((idx1, idx2))
        
            node_index += layer1_size
        
        # Colors based on authenticity
        if authenticity == "AUTHENTIC MEDIA":
            edge_color = '#00FF88'
            node_color = '#00D870'
        else:
            edge_color = '#FF3A00'
            node_color = '#D83000'
            
        # Plot connections first (edges)
        for start, end in connections:
            if np.random.random() < 0.3:  # Only draw some connections for clarity
                ax.plot3D(
                    [nodes[start][0], nodes[end][0]],
                    [nodes[start][1], nodes[end][1]],
                    [nodes[start][2], nodes[end][2]],
                    alpha=0.3, linewidth=0.5, color=edge_color
                )
        
        # Plot neurons (nodes)
        ax.scatter(
            nodes[:, 0], nodes[:, 1], nodes[:, 2],
            c=[node_color] * len(nodes),
            s=50 * (0.1 + 0.9 * np.random.random(len(nodes))),  # Varying sizes
            alpha=0.7, edgecolors='none'
        )
            
        # Set consistent viewpoint
        ax.view_init(elev=20, azim=30)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-3.5, 3.5)
        
        # Add title
        authenticity_color = '#00FF88' if authenticity == "AUTHENTIC MEDIA" else '#FF3A00'
        title = f"{'AUTHENTIC MEDIA' if authenticity == 'AUTHENTIC MEDIA' else 'MANIPULATED MEDIA'}"
        subtitle = f"Confidence: {confidence}%"
        
        ax.text2D(0.5, 0.95, title, transform=ax.transAxes, 
                 fontsize=14, fontweight='bold', ha='center', color=authenticity_color)
        ax.text2D(0.5, 0.9, subtitle, transform=ax.transAxes, 
                 fontsize=12, ha='center', color='white', alpha=0.8)
        
        return fig
    
    def create_animation(self, frames=30, fps=15, authenticity="AUTHENTIC MEDIA", confidence=85, output_path=None):
        """Create a neural network animation"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set background color
        fig.patch.set_facecolor('#0A1128')
        ax.set_facecolor('#0A1128')
        
        # Remove axes and grid
        ax.set_axis_off()
        
        # Create network layers
        layer_sizes = [8, 16, 32, 16, 8, 4, 1]
        layer_positions = np.linspace(-3, 3, len(layer_sizes))
        
        # Generate node positions
        nodes = []
        for i, size in enumerate(layer_sizes):
            z = layer_positions[i]
            for j in range(size):
                angle = j * 2 * np.pi / size
                r = 1.5 * np.sqrt(size) / 4  # Radius based on layer size
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                nodes.append((x, y, z))
        
        nodes = np.array(nodes)
        
        # Generate connections between adjacent layers
        connections = []
        node_index = 0
        for i in range(len(layer_sizes) - 1):
            layer1_size = layer_sizes[i]
            layer2_size = layer_sizes[i + 1]
            
            for j in range(layer1_size):
                idx1 = node_index + j
                for k in range(layer2_size):
                    idx2 = node_index + layer1_size + k
                    connections.append((idx1, idx2))
        
            node_index += layer1_size
        
        # Colors based on authenticity
        if authenticity == "AUTHENTIC MEDIA":
            edge_color = '#00FF88'
            node_color = '#00D870'
        else:
            edge_color = '#FF3A00'
            node_color = '#D83000'
            
        # Set title
        authenticity_color = '#00FF88' if authenticity == "AUTHENTIC MEDIA" else '#FF3A00'
        title = f"{'AUTHENTIC MEDIA' if authenticity == 'AUTHENTIC MEDIA' else 'MANIPULATED MEDIA'}"
        subtitle = f"Confidence: {confidence}%"
        
        ax.text2D(0.5, 0.95, title, transform=ax.transAxes, 
                 fontsize=14, fontweight='bold', ha='center', color=authenticity_color)
        ax.text2D(0.5, 0.9, subtitle, transform=ax.transAxes, 
                 fontsize=12, ha='center', color='white', alpha=0.8)
        
        # Plot connections (edges)
        edge_plots = []
        for start, end in connections:
            if np.random.random() < 0.3:  # Only draw some connections for clarity
                line, = ax.plot3D(
                    [nodes[start][0], nodes[end][0]],
                    [nodes[start][1], nodes[end][1]],
                    [nodes[start][2], nodes[end][2]],
                    alpha=0.3, linewidth=0.5, color=edge_color
                )
                edge_plots.append(line)
        
        # Plot neurons (nodes)
        scatter = ax.scatter(
            nodes[:, 0], nodes[:, 1], nodes[:, 2],
            c=[node_color] * len(nodes),
            s=50 * (0.1 + 0.9 * np.random.random(len(nodes))),  # Varying sizes
            alpha=0.7, edgecolors='none'
        )
        
        # Set consistent viewpoint
        ax.view_init(elev=20, azim=30)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-3.5, 3.5)
        
        # Animation update function
        def update(frame):
            # Rotate the view
            ax.view_init(elev=20, azim=30 + frame * 3)
            
            # Pulse the nodes by changing their size
            time_factor = frame / frames
            sizes = 50 * (0.3 + 0.7 * (0.5 + 0.5 * np.sin(time_factor * 2 * np.pi + np.arange(len(nodes)))))
            scatter._sizes = sizes
            
            # Pulse the edges
            for i, (start, end) in enumerate(connections):
                if i < len(edge_plots):
                    alpha = 0.1 + 0.3 * (0.5 + 0.5 * np.sin(time_factor * 2 * np.pi + i * 0.1))
                    edge_plots[i].set_alpha(alpha)
            
            return [scatter] + edge_plots
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=1000/fps, blit=True
        )
        
        if output_path:
            # Save as video or GIF
            try:
                if output_path.endswith('.mp4'):
                    writer = animation.FFMpegWriter(fps=fps)
                    anim.save(output_path, writer=writer)
                else:
                    anim.save(output_path, writer='pillow', fps=fps)
                return output_path
            except Exception as e:
                logger.error(f"Error saving animation: {e}")
                # Fallback: save a single frame
                return self.save_frame(fig, output_path)
        else:
            # Return a sequence of base64 encoded frames
            frames_base64 = []
            for i in range(frames):
                update(i)
                frame_base64 = self.save_frame(fig)
                frames_base64.append(frame_base64)
            
            plt.close(fig)
            return frames_base64


class BlockchainAnimation(AnimationGenerator):
    """Create 3D blockchain animation for SatyaChain™"""
    
    def __init__(self, size=(800, 600), dpi=100):
        super().__init__(size, dpi)
    
    def create_static_frame(self, is_verified=True, verification_progress=100):
        """Create a static blockchain visualization"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set dark background
        fig.patch.set_facecolor('#10172A')
        ax.set_facecolor('#10172A')
        
        # Remove axes and grid
        ax.set_axis_off()
        
        # Create blockchain
        num_blocks = 12
        block_positions = np.zeros((num_blocks, 3))
        
        # Position blocks in a helix
        t = np.linspace(0, 4 * np.pi, num_blocks)
        radius = 3
        block_positions[:, 0] = radius * np.cos(t)  # x
        block_positions[:, 1] = radius * np.sin(t)  # y
        block_positions[:, 2] = np.linspace(-3, 3, num_blocks)  # z
        
        # Colors based on verification status
        if is_verified:
            block_color = '#22cc88'
            edge_color = '#00aa66'
        else:
            block_color = '#cc5522'
            edge_color = '#aa3300'
            
        # Determine how many blocks to highlight based on progress
        progress_blocks = int(np.ceil(num_blocks * verification_progress / 100))
        
        # Plot connections (chain links)
        for i in range(num_blocks - 1):
            if i < progress_blocks - 1:
                color = edge_color
                alpha = 0.8
            else:
                color = '#4a8dff'
                alpha = 0.4
                
            ax.plot3D(
                block_positions[i:i+2, 0],
                block_positions[i:i+2, 1],
                block_positions[i:i+2, 2],
                color=color, alpha=alpha, linewidth=2
            )
        
        # Plot blockchain blocks
        for i in range(num_blocks):
            if i < progress_blocks:
                color = block_color
                alpha = 0.9
            else:
                color = '#2a5db0'
                alpha = 0.6
                
            ax.scatter(
                block_positions[i, 0],
                block_positions[i, 1],
                block_positions[i, 2],
                color=color, alpha=alpha,
                s=300, edgecolors='white', linewidths=1
            )
            
            # Add hash-like text for each block
            hash_text = f"#{hex(i + 0x1000)[2:]}"
            ax.text(
                block_positions[i, 0], 
                block_positions[i, 1], 
                block_positions[i, 2] + 0.2, 
                hash_text, 
                color='white', 
                fontsize=8, 
                ha='center', 
                va='center'
            )
        
        # Set viewpoint
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        
        # Add title
        verification_color = '#22cc88' if is_verified else '#cc5522'
        if verification_progress < 100:
            title = f"SatyaChain™ Verification in Progress"
            subtitle = f"Progress: {verification_progress}%"
        else:
            status = "VERIFIED" if is_verified else "FAILED"
            title = f"SatyaChain™ Verification: {status}"
            subtitle = "Blockchain verification complete"
            
        ax.text2D(0.5, 0.95, title, transform=ax.transAxes, 
                 fontsize=14, fontweight='bold', ha='center', color=verification_color)
        ax.text2D(0.5, 0.9, subtitle, transform=ax.transAxes, 
                 fontsize=12, ha='center', color='white', alpha=0.8)
        
        return fig
    
    def create_animation(self, frames=30, fps=15, is_verified=True, output_path=None):
        """Create a blockchain animation"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set dark background
        fig.patch.set_facecolor('#10172A')
        ax.set_facecolor('#10172A')
        
        # Remove axes and grid
        ax.set_axis_off()
        
        # Create blockchain
        num_blocks = 12
        block_positions = np.zeros((num_blocks, 3))
        
        # Position blocks in a helix
        t = np.linspace(0, 4 * np.pi, num_blocks)
        radius = 3
        block_positions[:, 0] = radius * np.cos(t)  # x
        block_positions[:, 1] = radius * np.sin(t)  # y
        block_positions[:, 2] = np.linspace(-3, 3, num_blocks)  # z
        
        # Colors based on verification status
        if is_verified:
            verified_block_color = '#22cc88'
            verified_edge_color = '#00aa66'
        else:
            verified_block_color = '#cc5522'
            verified_edge_color = '#aa3300'
        
        pending_block_color = '#2a5db0'
        pending_edge_color = '#4a8dff'
        
        # Plot initial connections (chain links)
        lines = []
        for i in range(num_blocks - 1):
            line, = ax.plot3D(
                block_positions[i:i+2, 0],
                block_positions[i:i+2, 1],
                block_positions[i:i+2, 2],
                color=pending_edge_color, alpha=0.4, linewidth=2
            )
            lines.append(line)
        
        # Plot initial blockchain blocks
        blocks = ax.scatter(
            block_positions[:, 0],
            block_positions[:, 1],
            block_positions[:, 2],
            color=pending_block_color, alpha=0.6,
            s=300, edgecolors='white', linewidths=1
        )
        
        # Add hash-like text for each block
        texts = []
        for i in range(num_blocks):
            hash_text = f"#{hex(i + 0x1000)[2:]}"
            text = ax.text(
                block_positions[i, 0], 
                block_positions[i, 1], 
                block_positions[i, 2] + 0.2, 
                hash_text, 
                color='white', 
                fontsize=8, 
                ha='center', 
                va='center'
            )
            texts.append(text)
        
        # Set viewpoint
        ax.view_init(elev=30, azim=45)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        
        # Add title
        verification_color = '#22cc88' if is_verified else '#cc5522'
        title_obj = ax.text2D(0.5, 0.95, "SatyaChain™ Verification in Progress", 
                             transform=ax.transAxes, fontsize=14, fontweight='bold', 
                             ha='center', color=verification_color)
        subtitle_obj = ax.text2D(0.5, 0.9, "Progress: 0%", 
                                transform=ax.transAxes, fontsize=12, 
                                ha='center', color='white', alpha=0.8)
        
        # Animation update function
        def update(frame):
            # Rotate the view
            ax.view_init(elev=30, azim=45 + frame * 2)
            
            # Calculate progress
            progress = min(100, int((frame / (frames - 1)) * 100))
            progress_blocks = int(np.ceil(num_blocks * progress / 100))
            
            # Update blocks colors
            colors = []
            sizes = []
            for i in range(num_blocks):
                if i < progress_blocks:
                    colors.append(verified_block_color if is_verified else verified_block_color)
                    sizes.append(300 * (1.0 + 0.1 * np.sin(frame * 0.2 + i * 0.5)))
                else:
                    colors.append(pending_block_color)
                    sizes.append(300)
            
            # Update block colors and sizes
            blocks._facecolor3d = colors
            blocks._sizes = sizes
            
            # Update chain links
            for i, line in enumerate(lines):
                if i < progress_blocks - 1:
                    line.set_color(verified_edge_color if is_verified else verified_edge_color)
                    line.set_alpha(0.8)
                else:
                    line.set_color(pending_edge_color)
                    line.set_alpha(0.4)
            
            # Update title
            if progress < 100:
                title_obj.set_text(f"SatyaChain™ Verification in Progress")
                subtitle_obj.set_text(f"Progress: {progress}%")
            else:
                status = "VERIFIED" if is_verified else "FAILED"
                title_obj.set_text(f"SatyaChain™ Verification: {status}")
                subtitle_obj.set_text("Blockchain verification complete")
            
            return [blocks, title_obj, subtitle_obj] + lines + texts
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=1000/fps, blit=True
        )
        
        if output_path:
            # Save as video or GIF
            try:
                if output_path.endswith('.mp4'):
                    writer = animation.FFMpegWriter(fps=fps)
                    anim.save(output_path, writer=writer)
                else:
                    anim.save(output_path, writer='pillow', fps=fps)
                return output_path
            except Exception as e:
                logger.error(f"Error saving animation: {e}")
                # Fallback: save a single frame
                return self.save_frame(fig, output_path)
        else:
            # Return a sequence of base64 encoded frames
            frames_base64 = []
            for i in range(frames):
                update(i)
                frame_base64 = self.save_frame(fig)
                frames_base64.append(frame_base64)
            
            plt.close(fig)
            return frames_base64


class WaveformAnimation(AnimationGenerator):
    """Create audio waveform animation for multi-language lip-sync analysis"""
    
    def __init__(self, size=(800, 600), dpi=100, language="english"):
        super().__init__(size, dpi)
        self.language = language
        
        # Color themes for different languages
        self.color_themes = {
            "english": {
                "primary": "#2979ff",
                "secondary": "#00d4ff",
                "background": "#0a1428",
                "text": "#ffffff"
            },
            "hindi": {
                "primary": "#ff2979",
                "secondary": "#ff0080",
                "background": "#280a14",
                "text": "#ffffff"
            },
            "tamil": {
                "primary": "#79ff29",
                "secondary": "#adff00",
                "background": "#14280a",
                "text": "#ffffff"
            }
        }
        
        # Use english as fallback
        if language not in self.color_themes:
            self.language = "english"
    
    def create_static_frame(self, sync_score=85):
        """Create a static waveform visualization"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        
        # Get color theme
        theme = self.color_themes[self.language]
        
        # Set background color
        fig.patch.set_facecolor(theme["background"])
        
        # Create grid for waveform and mouth visualization
        grid = plt.GridSpec(3, 1, height_ratios=[1, 2, 1])
        
        # Top section - title and language indicator
        ax_top = fig.add_subplot(grid[0])
        ax_top.set_facecolor(theme["background"])
        ax_top.set_xlim(0, 10)
        ax_top.set_ylim(0, 1)
        ax_top.axis('off')
        
        # Add title
        ax_top.text(5, 0.6, f"{self.language.upper()} LIP-SYNC ANALYZER", 
                   fontsize=16, fontweight='bold', color=theme["text"],
                   ha='center', va='center')
        
        # Add language indicator
        ax_top.scatter(3, 0.6, color=theme["primary"], s=100, edgecolors='white', linewidths=1)
        
        # Middle section - waveform
        ax_mid = fig.add_subplot(grid[1])
        ax_mid.set_facecolor(theme["background"])
        ax_mid.set_xlim(0, 100)
        ax_mid.set_ylim(-1, 1)
        ax_mid.axis('off')
        
        # Generate waveform data
        x = np.linspace(0, 100, 1000)
        y1 = 0.7 * np.sin(x * 0.2) + 0.3 * np.sin(x * 0.5)
        y2 = 0.5 * np.sin(x * 0.3) + 0.2 * np.sin(x * 0.7)
        y3 = 0.3 * np.sin(x * 0.4) + 0.1 * np.sin(x * 0.9)
        
        # Plot multiple waveforms
        ax_mid.fill_between(x, y1, alpha=0.3, color=theme["secondary"])
        ax_mid.fill_between(x, y2, alpha=0.2, color=theme["secondary"])
        ax_mid.plot(x, y1, color=theme["primary"], alpha=0.8, linewidth=2)
        ax_mid.plot(x, y2, color=theme["primary"], alpha=0.6, linewidth=1.5)
        ax_mid.plot(x, y3, color=theme["primary"], alpha=0.4, linewidth=1)
        
        # Bottom section - phoneme visualization and score
        ax_bot = fig.add_subplot(grid[2])
        ax_bot.set_facecolor(theme["background"])
        ax_bot.set_xlim(0, 10)
        ax_bot.set_ylim(0, 1)
        ax_bot.axis('off')
        
        # Get phonemes based on language
        if self.language == "english":
            phonemes = ["AA", "AE", "AH", "AO", "EH", "IH", "IY", "UH"]
        elif self.language == "hindi":
            phonemes = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ"]
        else:  # tamil
            phonemes = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ"]
            
        # Plot phoneme indicators
        phoneme_positions = np.linspace(1, 9, len(phonemes))
        for i, (pos, phoneme) in enumerate(zip(phoneme_positions, phonemes)):
            # Random activation for visual interest
            is_active = np.random.random() > 0.7
            color = theme["primary"] if is_active else 'gray'
            alpha = 0.9 if is_active else 0.4
            
            ax_bot.scatter(pos, 0.7, color=color, alpha=alpha, s=120, edgecolors='white', linewidths=1)
            ax_bot.text(pos, 0.7, phoneme, color='white', ha='center', va='center', fontsize=9)
        
        # Add sync score
        score_color = '#22cc88' if sync_score >= 70 else '#cc5522'
        ax_bot.text(5, 0.2, f"Sync Confidence: {sync_score}%", 
                   fontsize=14, fontweight='bold', color=score_color,
                   ha='center', va='center')
        
        return fig
    
    def create_animation(self, frames=30, fps=15, sync_score=85, output_path=None):
        """Create a waveform animation"""
        fig = plt.figure(figsize=(self.size[0]/self.dpi, self.size[1]/self.dpi), dpi=self.dpi)
        
        # Get color theme
        theme = self.color_themes[self.language]
        
        # Set background color
        fig.patch.set_facecolor(theme["background"])
        
        # Create grid for waveform and mouth visualization
        grid = plt.GridSpec(3, 1, height_ratios=[1, 2, 1])
        
        # Top section - title and language indicator
        ax_top = fig.add_subplot(grid[0])
        ax_top.set_facecolor(theme["background"])
        ax_top.set_xlim(0, 10)
        ax_top.set_ylim(0, 1)
        ax_top.axis('off')
        
        # Add title
        title = ax_top.text(5, 0.6, f"{self.language.upper()} LIP-SYNC ANALYZER", 
                           fontsize=16, fontweight='bold', color=theme["text"],
                           ha='center', va='center')
        
        # Add language indicator
        indicator = ax_top.scatter(3, 0.6, color=theme["primary"], s=100, edgecolors='white', linewidths=1)
        
        # Middle section - waveform
        ax_mid = fig.add_subplot(grid[1])
        ax_mid.set_facecolor(theme["background"])
        ax_mid.set_xlim(0, 100)
        ax_mid.set_ylim(-1, 1)
        ax_mid.axis('off')
        
        # Generate initial waveform data
        x = np.linspace(0, 100, 1000)
        y1 = 0.7 * np.sin(x * 0.2) + 0.3 * np.sin(x * 0.5)
        y2 = 0.5 * np.sin(x * 0.3) + 0.2 * np.sin(x * 0.7)
        y3 = 0.3 * np.sin(x * 0.4) + 0.1 * np.sin(x * 0.9)
        
        # Plot multiple waveforms
        fill1 = ax_mid.fill_between(x, y1, alpha=0.3, color=theme["secondary"])
        fill2 = ax_mid.fill_between(x, y2, alpha=0.2, color=theme["secondary"])
        line1, = ax_mid.plot(x, y1, color=theme["primary"], alpha=0.8, linewidth=2)
        line2, = ax_mid.plot(x, y2, color=theme["primary"], alpha=0.6, linewidth=1.5)
        line3, = ax_mid.plot(x, y3, color=theme["primary"], alpha=0.4, linewidth=1)
        
        # Bottom section - phoneme visualization and score
        ax_bot = fig.add_subplot(grid[2])
        ax_bot.set_facecolor(theme["background"])
        ax_bot.set_xlim(0, 10)
        ax_bot.set_ylim(0, 1)
        ax_bot.axis('off')
        
        # Get phonemes based on language
        if self.language == "english":
            phonemes = ["AA", "AE", "AH", "AO", "EH", "IH", "IY", "UH"]
        elif self.language == "hindi":
            phonemes = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ"]
        else:  # tamil
            phonemes = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ"]
            
        # Plot phoneme indicators and text
        phoneme_positions = np.linspace(1, 9, len(phonemes))
        phoneme_dots = []
        phoneme_texts = []
        
        for i, (pos, phoneme) in enumerate(zip(phoneme_positions, phonemes)):
            # Initial state (all inactive)
            dot = ax_bot.scatter(pos, 0.7, color='gray', alpha=0.4, s=120, edgecolors='white', linewidths=1)
            text = ax_bot.text(pos, 0.7, phoneme, color='white', ha='center', va='center', fontsize=9)
            
            phoneme_dots.append(dot)
            phoneme_texts.append(text)
        
        # Add sync score
        score_color = '#22cc88' if sync_score >= 70 else '#cc5522'
        score_text = ax_bot.text(5, 0.2, f"Sync Confidence: {sync_score}%", 
                                fontsize=14, fontweight='bold', color=score_color,
                                ha='center', va='center')
        
        # Animation update function
        def update(frame):
            time_factor = frame / frames
            
            # Update waveform
            # Apply phase shift based on frame for animation
            shift = time_factor * 10
            y1_new = 0.7 * np.sin(x * 0.2 + shift) + 0.3 * np.sin(x * 0.5 + shift * 1.5)
            y2_new = 0.5 * np.sin(x * 0.3 + shift) + 0.2 * np.sin(x * 0.7 + shift * 1.5)
            y3_new = 0.3 * np.sin(x * 0.4 + shift) + 0.1 * np.sin(x * 0.9 + shift * 1.5)
            
            # Update line data
            line1.set_ydata(y1_new)
            line2.set_ydata(y2_new)
            line3.set_ydata(y3_new)
            
            # Can't update fill_between directly, so we clear and redraw
            fill1.remove()
            fill2.remove()
            fill1 = ax_mid.fill_between(x, y1_new, alpha=0.3, color=theme["secondary"])
            fill2 = ax_mid.fill_between(x, y2_new, alpha=0.2, color=theme["secondary"])
            
            # Update phoneme indicators
            # Randomly activate different phonemes for each frame
            active_indices = np.random.choice(
                len(phonemes), size=2, replace=False
            )
            
            for i, dot in enumerate(phoneme_dots):
                # Reset all dots to inactive
                dot.set_color('gray')
                dot.set_alpha(0.4)
                
                # Activate selected dots
                if i in active_indices:
                    dot.set_color(theme["primary"])
                    dot.set_alpha(0.9)
            
            # Pulse the language indicator
            indicator_size = 100 + 20 * np.sin(time_factor * 2 * np.pi)
            indicator.set_sizes([indicator_size])
            
            return [line1, line2, line3, indicator, score_text] + phoneme_dots + phoneme_texts
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=1000/fps, blit=True
        )
        
        if output_path:
            # Save as video or GIF
            try:
                if output_path.endswith('.mp4'):
                    writer = animation.FFMpegWriter(fps=fps)
                    anim.save(output_path, writer=writer)
                else:
                    anim.save(output_path, writer='pillow', fps=fps)
                return output_path
            except Exception as e:
                logger.error(f"Error saving animation: {e}")
                # Fallback: save a single frame
                return self.save_frame(fig, output_path)
        else:
            # Return a sequence of base64 encoded frames
            frames_base64 = []
            for i in range(frames):
                update(i)
                frame_base64 = self.save_frame(fig)
                frames_base64.append(frame_base64)
            
            plt.close(fig)
            return frames_base64


# Test and generate example animations
if __name__ == "__main__":
    output_dir = "animations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Neural network animation
    print("Creating neural network animation...")
    nn_anim = NeuralNetworkAnimation()
    nn_fig = nn_anim.create_static_frame(authenticity="AUTHENTIC MEDIA", confidence=85)
    nn_anim.save_frame(nn_fig, f"{output_dir}/neural_network.png")
    plt.close(nn_fig)
    
    # Blockchain animation
    print("Creating blockchain animation...")
    bc_anim = BlockchainAnimation()
    bc_fig = bc_anim.create_static_frame(is_verified=True, verification_progress=100)
    bc_anim.save_frame(bc_fig, f"{output_dir}/blockchain.png")
    plt.close(bc_fig)
    
    # Waveform animation for different languages
    for language in ["english", "hindi", "tamil"]:
        print(f"Creating {language} waveform animation...")
        wf_anim = WaveformAnimation(language=language)
        wf_fig = wf_anim.create_static_frame(sync_score=85)
        wf_anim.save_frame(wf_fig, f"{output_dir}/waveform_{language}.png")
        plt.close(wf_fig)
    
    print("All animations created successfully!")