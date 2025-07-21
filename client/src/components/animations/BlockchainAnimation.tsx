import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface BlockchainAnimationProps {
  isVerifying?: boolean;
  isVerified?: boolean;
}

const BlockchainAnimation: React.FC<BlockchainAnimationProps> = ({
  isVerifying = false,
  isVerified = false
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const frameIdRef = useRef<number>(0);
  const blocksRef = useRef<THREE.Mesh[]>([]);
  const linksRef = useRef<THREE.Line[]>([]);
  const timeRef = useRef<number>(0);
  
  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Create scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    
    // Create camera
    const camera = new THREE.PerspectiveCamera(
      60, 
      containerRef.current.clientWidth / containerRef.current.clientHeight, 
      0.1, 
      1000
    );
    camera.position.z = 20;
    cameraRef.current = camera;
    
    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setClearColor(0x000000, 0); // Transparent background
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Create lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);
    
    // Create blockchain visualization
    const createBlockchain = () => {
      const blockCount = 12;
      const blocks: THREE.Mesh[] = [];
      const links: THREE.Line[] = [];
      
      // Block material
      const blockMaterial = new THREE.MeshPhongMaterial({
        color: 0x2a5db0,
        emissive: 0x072d5e,
        transparent: true,
        opacity: 0.9,
        shininess: 90
      });
      
      // Create blocks
      for (let i = 0; i < blockCount; i++) {
        // Create block geometry
        const blockGeometry = new THREE.BoxGeometry(1.5, 1, 0.5);
        const block = new THREE.Mesh(blockGeometry, blockMaterial);
        
        // Position blocks in a helix pattern
        const angle = (i / blockCount) * Math.PI * 4; // 2 full rotations
        const radius = 5;
        block.position.x = Math.cos(angle) * radius;
        block.position.y = (i - blockCount / 2) * 1.2;
        block.position.z = Math.sin(angle) * radius;
        
        // Rotate blocks to face outward
        block.rotation.x = Math.PI / 6;
        block.rotation.y = angle + Math.PI / 2;
        
        scene.add(block);
        blocks.push(block);
        
        // Create connection lines between blocks
        if (i > 0) {
          const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            blocks[i-1].position,
            block.position
          ]);
          
          const lineMaterial = new THREE.LineBasicMaterial({ 
            color: 0x4a8dff,
            transparent: true,
            opacity: 0.7
          });
          
          const line = new THREE.Line(lineGeometry, lineMaterial);
          scene.add(line);
          links.push(line);
        }
      }
      
      blocksRef.current = blocks;
      linksRef.current = links;
    };
    
    createBlockchain();
    
    // Animation loop
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      timeRef.current += 0.01;
      
      // Rotate the entire blockchain
      if (blocksRef.current.length > 0) {
        blocksRef.current.forEach((block, index) => {
          const angle = (index / blocksRef.current.length) * Math.PI * 4 + timeRef.current * 0.3;
          const radius = 5;
          
          block.position.x = Math.cos(angle) * radius;
          block.position.z = Math.sin(angle) * radius;
          
          // Update block rotation to face outward
          block.rotation.y = angle + Math.PI / 2;
          
          // Add slight pulsing effect
          const pulseFactor = Math.sin(timeRef.current * 2 + index * 0.2) * 0.05 + 1;
          block.scale.set(pulseFactor, pulseFactor, pulseFactor);
        });
        
        // Update connection lines
        linksRef.current.forEach((line, index) => {
          if (index < blocksRef.current.length - 1) {
            const positions = line.geometry.attributes.position as THREE.BufferAttribute;
            positions.setXYZ(0, 
              blocksRef.current[index].position.x,
              blocksRef.current[index].position.y,
              blocksRef.current[index].position.z
            );
            positions.setXYZ(1,
              blocksRef.current[index + 1].position.x,
              blocksRef.current[index + 1].position.y,
              blocksRef.current[index + 1].position.z
            );
            positions.needsUpdate = true;
          }
        });
      }
      
      // Rotate camera slowly around the scene
      if (cameraRef.current) {
        const camRadius = 20;
        const camSpeed = 0.1;
        cameraRef.current.position.x = Math.cos(timeRef.current * camSpeed) * camRadius;
        cameraRef.current.position.z = Math.sin(timeRef.current * camSpeed) * camRadius;
        cameraRef.current.lookAt(new THREE.Vector3(0, 0, 0));
      }
      
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
    if (blocksRef.current.length === 0 || linksRef.current.length === 0) return;
    
    // Update block colors and animation speed based on state
    blocksRef.current.forEach((block, index) => {
      const material = block.material as THREE.MeshPhongMaterial;
      
      if (isVerifying) {
        // Verification in progress - show sequence of validation
        const activeIndex = Math.floor(Date.now() / 300) % blocksRef.current.length;
        if (index === activeIndex) {
          material.color.set(0xffaa00); // Highlight active block
          material.emissive.set(0x553300);
          material.opacity = 1;
        } else if (index < activeIndex) {
          material.color.set(0x22cc88); // Verified blocks
          material.emissive.set(0x115533);
          material.opacity = 0.9;
        } else {
          material.color.set(0x2a5db0); // Pending blocks
          material.emissive.set(0x072d5e);
          material.opacity = 0.7;
        }
      } else if (isVerified) {
        // Verified - all green
        material.color.set(0x22cc88);
        material.emissive.set(0x115533);
        material.opacity = 0.9;
      } else {
        // Default state - blue
        material.color.set(0x2a5db0);
        material.emissive.set(0x072d5e);
        material.opacity = 0.7;
      }
    });
    
    // Update link colors
    linksRef.current.forEach((link, index) => {
      const material = link.material as THREE.LineBasicMaterial;
      
      if (isVerifying) {
        const activeIndex = Math.floor(Date.now() / 300) % blocksRef.current.length;
        if (index < activeIndex) {
          material.color.set(0x22cc88); // Verified links
        } else {
          material.color.set(0x4a8dff); // Pending links
        }
      } else if (isVerified) {
        material.color.set(0x22cc88); // All verified
      } else {
        material.color.set(0x4a8dff); // Default
      }
    });
    
  }, [isVerifying, isVerified]);
  
  return (
    <div 
      ref={containerRef} 
      className="w-full h-64 md:h-80 relative rounded-lg overflow-hidden"
      style={{ background: 'linear-gradient(to bottom, rgba(15,25,50,0.9), rgba(5,10,30,0.95))' }}
    >
      {isVerifying && (
        <div className="absolute bottom-0 left-0 right-0 p-4 text-center">
          <div className="bg-black/50 text-white text-sm py-2 px-4 rounded-full inline-block">
            <span className="inline-block w-2 h-2 bg-yellow-500 rounded-full mr-2 animate-pulse"></span>
            Verifying on SatyaChain™ Blockchain
          </div>
        </div>
      )}
      
      {isVerified && (
        <div className="absolute bottom-0 left-0 right-0 p-4 text-center">
          <div className="bg-black/50 text-green-400 text-sm py-2 px-4 rounded-full inline-block">
            <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-2"></span>
            Verified on SatyaChain™ Blockchain
          </div>
        </div>
      )}
    </div>
  );
};

export default BlockchainAnimation;