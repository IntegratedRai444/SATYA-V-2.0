import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
<<<<<<< HEAD
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
=======
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f

interface ModelAnimationProps {
  isAnalyzing?: boolean;
  isAuthentic?: boolean;
}

const ModelAnimation: React.FC<ModelAnimationProps> = ({ 
  isAnalyzing = false,
  isAuthentic = false
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const frameIdRef = useRef<number>(0);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  
  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Create scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    
    // Create camera
    const camera = new THREE.PerspectiveCamera(
      75, 
      containerRef.current.clientWidth / containerRef.current.clientHeight, 
      0.1, 
      1000
    );
    camera.position.z = 5;
    cameraRef.current = camera;
    
    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setClearColor(0x000000, 0); // Transparent background
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Add controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enableZoom = false;
    
    // Create light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);
    
    // Create neural network-like structure for animation
    const createNeuralNetwork = () => {
      // Create points for nodes
      const nodeCount = 100;
      const nodeGeometry = new THREE.BufferGeometry();
      const nodePositions = new Float32Array(nodeCount * 3);
      const nodeSizes = new Float32Array(nodeCount);
      
      for (let i = 0; i < nodeCount; i++) {
        // Position in a brain-like shape
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;
        const radius = 2 + Math.random() * 0.5;
        
        nodePositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
        nodePositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        nodePositions[i * 3 + 2] = radius * Math.cos(phi);
        
        // Random sizes
        nodeSizes[i] = Math.random() * 0.1 + 0.02;
      }
      
      nodeGeometry.setAttribute('position', new THREE.BufferAttribute(nodePositions, 3));
      nodeGeometry.setAttribute('size', new THREE.BufferAttribute(nodeSizes, 1));
      
      // Create material for nodes
      const nodeMaterial = new THREE.PointsMaterial({
        color: 0x00aaff,
        size: 0.1,
        transparent: true,
        opacity: 0.7,
        sizeAttenuation: true
      });
      
      // Create points mesh
      const points = new THREE.Points(nodeGeometry, nodeMaterial);
      scene.add(points);
      pointsRef.current = points;
      
      // Create face mesh
      const faceGeometry = new THREE.SphereGeometry(1.8, 32, 32);
      const faceMaterial = new THREE.MeshStandardMaterial({
        color: 0x1a2a3a,
        wireframe: true,
        transparent: true,
        opacity: 0.5
      });
      
      const faceMesh = new THREE.Mesh(faceGeometry, faceMaterial);
      scene.add(faceMesh);
      meshRef.current = faceMesh;
    };
    
    createNeuralNetwork();
    
    // Animation loop
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      
      if (pointsRef.current) {
        pointsRef.current.rotation.y += 0.002;
      }
      
      if (meshRef.current) {
        meshRef.current.rotation.y += 0.001;
      }
      
      controls.update();
      renderer.render(scene, camera);
    };
    
    animate();
    
    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;
      
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;
      
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      
      rendererRef.current.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Cleanup
    return () => {
      if (containerRef.current && rendererRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }
      
      cancelAnimationFrame(frameIdRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, []);
  
  // Update animation based on props
  useEffect(() => {
    if (!meshRef.current || !pointsRef.current) return;
    
    // Update material colors and animation speed based on state
    if (isAnalyzing) {
      // Analyzing state - bright blue pulsing
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      material.color.set(0x0088ff);
      material.opacity = 0.7;
      
      const pointMaterial = pointsRef.current.material as THREE.PointsMaterial;
      pointMaterial.color.set(0x00aaff);
      pointMaterial.size = 0.1;
      
      // Faster rotation during analysis
      meshRef.current.rotation.y += 0.05;
    } else if (isAuthentic === true) {
      // Authentic state - green glow
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      material.color.set(0x00ff88);
      material.opacity = 0.6;
      
      const pointMaterial = pointsRef.current.material as THREE.PointsMaterial;
      pointMaterial.color.set(0x00ff88);
      pointMaterial.size = 0.12;
    } else if (isAuthentic === false) {
      // Deepfake state - red alert
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      material.color.set(0xff3300);
      material.opacity = 0.6;
      
      const pointMaterial = pointsRef.current.material as THREE.PointsMaterial;
      pointMaterial.color.set(0xff5500);
      pointMaterial.size = 0.12;
    }
  }, [isAnalyzing, isAuthentic]);
  
  return (
    <div 
      ref={containerRef} 
      className="w-full h-64 md:h-96 relative rounded-lg overflow-hidden"
      style={{ background: 'linear-gradient(to bottom, rgba(10,10,30,0.8), rgba(5,5,15,0.9))' }}
    >
      {isAnalyzing && (
        <div className="absolute inset-0 flex items-center justify-center z-10 text-white">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mb-2"></div>
            <p className="text-lg font-medium">Analyzing Media...</p>
            <p className="text-sm text-blue-300">Advanced AI models processing</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelAnimation;