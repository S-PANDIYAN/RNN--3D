/**
 * 3D Visualization Controller
 * Handles all Three.js rendering and animations for the RNN visualization
 */

class Visualization3D {
    constructor(container) {
        this.container = container;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.controls = null;
        
        // Animation properties
        this.animationSpeed = 1.0;
        this.isAnimating = false;
        this.animationQueue = [];
        
        // 3D Objects storage
        this.tokenObjects = [];
        this.embeddingObjects = [];
        this.rnnCells = [];
        this.hiddenStates = [];
        this.connectionLines = [];
        this.gradientArrows = [];
        
        // Colors
        this.colors = {
            token: 0x4ecdc4,
            embedding: 0xff6b6b,
            rnnCell: 0x6c5ce7,
            hiddenState: 0xfdcb6e,
            positive: 0x00b894,
            neutral: 0xfdcb6e,
            negative: 0xe17055,
            gradient: 0xff7675,
            connection: 0x74b9ff
        };
        
        this.setupRenderer();
        this.setupCamera();
        this.setupControls();
        this.setupLighting();
        this.setupEnvironment();
        
        // Start render loop
        this.animate();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    setupRenderer() {
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.setClearColor(0x000000, 0);
        this.container.appendChild(this.renderer.domElement);
    }
    
    setupCamera() {
        this.camera.position.set(15, 12, 25);
        this.camera.lookAt(0, 0, 0);
        this.camera.fov = 75;
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.near = 0.1;
        this.camera.far = 1000;
        this.camera.updateProjectionMatrix();
    }
    
    setupControls() {
        // Try multiple ways to access OrbitControls
        let OrbitControlsClass = null;
        
        if (typeof THREE.OrbitControls !== 'undefined') {
            OrbitControlsClass = THREE.OrbitControls;
        } else if (typeof OrbitControls !== 'undefined') {
            OrbitControlsClass = OrbitControls;
        } else if (window.THREE && window.THREE.OrbitControls) {
            OrbitControlsClass = window.THREE.OrbitControls;
        }
        
        if (OrbitControlsClass) {
            try {
                this.controls = new OrbitControlsClass(this.camera, this.renderer.domElement);
                
                // Configure controls for better user experience
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
                this.controls.screenSpacePanning = false;
                this.controls.minDistance = 8;
                this.controls.maxDistance = 100;
                this.controls.maxPolarAngle = Math.PI;
                this.controls.autoRotate = false;
                this.controls.enableZoom = true;
                this.controls.enableRotate = true;
                this.controls.enablePan = true;
                
                // Set initial target
                this.controls.target.set(0, 0, 0);
                this.controls.update();
                
                console.log('OrbitControls initialized successfully');
            } catch (error) {
                console.error('Failed to initialize OrbitControls:', error);
                this.setupEnhancedBasicControls();
            }
        } else {
            console.warn('OrbitControls not found, using enhanced basic camera controls');
            this.setupEnhancedBasicControls();
        }
    }
    
    setupEnhancedBasicControls() {
        // Enhanced fallback mouse controls with rotation around center
        let isMouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let cameraRadius = 25;
        let cameraTheta = 0; // horizontal angle
        let cameraPhi = Math.PI / 3; // vertical angle (60 degrees)
        
        const updateCameraPosition = () => {
            this.camera.position.x = cameraRadius * Math.sin(cameraPhi) * Math.cos(cameraTheta);
            this.camera.position.y = cameraRadius * Math.cos(cameraPhi);
            this.camera.position.z = cameraRadius * Math.sin(cameraPhi) * Math.sin(cameraTheta);
            this.camera.lookAt(0, 0, 0);
        };
        
        // Set initial camera position
        updateCameraPosition();
        
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            isMouseDown = true;
            mouseX = e.clientX;
            mouseY = e.clientY;
            this.renderer.domElement.style.cursor = 'grabbing';
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            isMouseDown = false;
            this.renderer.domElement.style.cursor = 'grab';
        });
        
        this.renderer.domElement.addEventListener('mouseleave', () => {
            isMouseDown = false;
            this.renderer.domElement.style.cursor = 'grab';
        });
        
        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (!isMouseDown) return;
            
            const deltaX = e.clientX - mouseX;
            const deltaY = e.clientY - mouseY;
            
            // Update angles based on mouse movement
            cameraTheta -= deltaX * 0.01;
            cameraPhi += deltaY * 0.01;
            
            // Clamp vertical angle to prevent flipping
            cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, cameraPhi));
            
            updateCameraPosition();
            
            mouseX = e.clientX;
            mouseY = e.clientY;
        });
        
        // Enhanced zoom with mouse wheel
        this.renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeed = 0.1;
            const delta = e.deltaY > 0 ? 1 + zoomSpeed : 1 - zoomSpeed;
            cameraRadius *= delta;
            
            // Clamp zoom distance
            cameraRadius = Math.max(8, Math.min(100, cameraRadius));
            
            updateCameraPosition();
        });
        
        // Set cursor style
        this.renderer.domElement.style.cursor = 'grab';
        
        console.log('Enhanced basic camera controls initialized');
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Main directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Colored accent lights
        const accentLight1 = new THREE.PointLight(0x00f5ff, 0.5, 30);
        accentLight1.position.set(-10, 5, 10);
        this.scene.add(accentLight1);
        
        const accentLight2 = new THREE.PointLight(0xff00ff, 0.5, 30);
        accentLight2.position.set(10, 5, -10);
        this.scene.add(accentLight2);
    }
    
    setupEnvironment() {
        // Create a grid
        const gridHelper = new THREE.GridHelper(40, 40, 0x444444, 0x222222);
        gridHelper.position.y = -5;
        this.scene.add(gridHelper);
        
        // Add some particle effects
        this.createParticleSystem();
    }
    
    createParticleSystem() {
        const particleCount = 200;
        const particles = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 100;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 100;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 100;
        }
        
        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            color: 0x888888,
            size: 0.1,
            transparent: true,
            opacity: 0.3
        });
        
        const particleSystem = new THREE.Points(particles, particleMaterial);
        this.scene.add(particleSystem);
    }
    
    /**
     * Clear all 3D objects from the scene
     */
    clearScene() {
        this.tokenObjects.forEach(obj => this.scene.remove(obj));
        this.embeddingObjects.forEach(obj => this.scene.remove(obj));
        this.rnnCells.forEach(obj => this.scene.remove(obj));
        this.hiddenStates.forEach(obj => this.scene.remove(obj));
        this.connectionLines.forEach(obj => this.scene.remove(obj));
        this.gradientArrows.forEach(obj => this.scene.remove(obj));
        
        this.tokenObjects = [];
        this.embeddingObjects = [];
        this.rnnCells = [];
        this.hiddenStates = [];
        this.connectionLines = [];
        this.gradientArrows = [];
    }
    
    /**
     * Visualize tokenization step
     */
    async visualizeTokenization(tokens) {
        this.clearScene();
        
        const tokenSpacing = 3;
        const startX = -(tokens.length - 1) * tokenSpacing / 2;
        
        for (let i = 0; i < tokens.length; i++) {
            const tokenObj = this.createTokenObject(tokens[i], i);
            tokenObj.position.set(startX + i * tokenSpacing, 5, 0);
            tokenObj.scale.set(0, 0, 0);
            
            this.scene.add(tokenObj);
            this.tokenObjects.push(tokenObj);
            
            // Animate token appearance
            await this.animateObjectScale(tokenObj, { x: 1, y: 1, z: 1 }, 500);
            await this.delay(200);
        }
    }
    
    /**
     * Visualize embedding step
     */
    async visualizeEmbedding(tokens, embeddings) {
        const embeddingSpacing = 3;
        const startX = -(tokens.length - 1) * embeddingSpacing / 2;
        
        for (let i = 0; i < embeddings.length; i++) {
            const embeddingObj = this.createEmbeddingObject(embeddings[i], i);
            embeddingObj.position.set(startX + i * embeddingSpacing, 2, 0);
            embeddingObj.scale.set(0, 0, 0);
            
            this.scene.add(embeddingObj);
            this.embeddingObjects.push(embeddingObj);
            
            // Animate embedding appearance
            await this.animateObjectScale(embeddingObj, { x: 1, y: 1, z: 1 }, 500);
            
            // Create connection line from token to embedding
            const connectionLine = this.createConnectionLine(
                this.tokenObjects[i].position,
                embeddingObj.position
            );
            this.scene.add(connectionLine);
            this.connectionLines.push(connectionLine);
            
            await this.delay(300);
        }
    }
    
    /**
     * Visualize RNN processing
     */
    async visualizeRNNProcessing(computationHistory) {
        const rnnSpacing = 4;
        const timeStepSpacing = 6;
        
        // Create RNN cells for each time step
        for (let t = 0; t < computationHistory.length; t++) {
            const rnnCell = this.createRNNCell(t);
            rnnCell.position.set(t * timeStepSpacing - (computationHistory.length - 1) * timeStepSpacing / 2, -2, 0);
            
            this.scene.add(rnnCell);
            this.rnnCells.push(rnnCell);
            
            // Create hidden state visualization
            const hiddenState = this.createHiddenStateObject(computationHistory[t].hiddenState, t);
            hiddenState.position.set(rnnCell.position.x, 0, -3);
            
            this.scene.add(hiddenState);
            this.hiddenStates.push(hiddenState);
            
            // Animate data flow
            await this.animateDataFlow(t, computationHistory[t]);
            await this.delay(800);
        }
    }
    
    /**
     * Visualize prediction output
     */
    async visualizePrediction(probabilities) {
        const sentimentLabels = ['Positive', 'Neutral', 'Negative'];
        const colors = [this.colors.positive, this.colors.neutral, this.colors.negative];
        
        for (let i = 0; i < probabilities.length; i++) {
            const outputObj = this.createOutputObject(probabilities[i], sentimentLabels[i], colors[i]);
            outputObj.position.set(i * 4 - 4, -5, 5);
            
            this.scene.add(outputObj);
            
            // Animate output appearance
            await this.animateObjectScale(outputObj, { x: 1, y: 1, z: 1 }, 500);
            await this.delay(200);
        }
    }
    
    /**
     * Visualize backpropagation
     */
    async visualizeBackpropagation(gradients) {
        // Create gradient arrows flowing backward
        for (let t = gradients.hiddenGradients.length - 1; t >= 0; t--) {
            const arrow = this.createGradientArrow(t);
            arrow.position.copy(this.rnnCells[t].position);
            arrow.position.y += 2;
            
            this.scene.add(arrow);
            this.gradientArrows.push(arrow);
            
            // Animate gradient flow
            await this.animateGradientFlow(arrow, t);
            await this.delay(400);
        }
    }
    
    /**
     * Create token 3D object
     */
    createTokenObject(token, index) {
        const geometry = new THREE.BoxGeometry(2, 0.5, 0.5);
        const material = new THREE.MeshLambertMaterial({ 
            color: this.colors.token,
            transparent: true,
            opacity: 0.8
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Add text label
        const textObj = this.createTextLabel(token, 0xffffff);
        textObj.position.y = 0.8;
        mesh.add(textObj);
        
        return mesh;
    }
    
    /**
     * Create embedding 3D object
     */
    createEmbeddingObject(embedding, index) {
        const geometry = new THREE.CylinderGeometry(0.3, 0.3, 2, 8);
        const material = new THREE.MeshLambertMaterial({ 
            color: this.colors.embedding,
            transparent: true,
            opacity: 0.7
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Add glow effect
        const glowGeometry = new THREE.CylinderGeometry(0.4, 0.4, 2.2, 8);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: this.colors.embedding,
            transparent: true,
            opacity: 0.2
        });
        const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
        mesh.add(glowMesh);
        
        return mesh;
    }
    
    /**
     * Create RNN cell 3D object
     */
    createRNNCell(timeStep) {
        const geometry = new THREE.SphereGeometry(1, 16, 16);
        const material = new THREE.MeshLambertMaterial({ 
            color: this.colors.rnnCell,
            transparent: true,
            opacity: 0.8
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Add text label
        const textObj = this.createTextLabel(`RNN ${timeStep + 1}`, 0xffffff);
        textObj.position.y = 1.5;
        mesh.add(textObj);
        
        return mesh;
    }
    
    /**
     * Create hidden state visualization
     */
    createHiddenStateObject(hiddenState, timeStep) {
        const geometry = new THREE.IcosahedronGeometry(0.8, 1);
        const material = new THREE.MeshLambertMaterial({ 
            color: this.colors.hiddenState,
            transparent: true,
            opacity: 0.7
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Add pulsing animation
        mesh.userData = { 
            timeStep: timeStep,
            originalScale: 1,
            pulsePhase: 0
        };
        
        return mesh;
    }
    
    /**
     * Create output visualization object
     */
    createOutputObject(probability, label, color) {
        const height = probability * 3 + 0.5;
        const geometry = new THREE.CylinderGeometry(0.5, 0.5, height, 8);
        const material = new THREE.MeshLambertMaterial({ 
            color: color,
            transparent: true,
            opacity: 0.8
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.y = height / 2;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Add text label
        const textObj = this.createTextLabel(`${label}\n${(probability * 100).toFixed(1)}%`, 0xffffff);
        textObj.position.y = height + 0.5;
        mesh.add(textObj);
        
        const container = new THREE.Group();
        container.add(mesh);
        container.scale.set(0, 0, 0);
        
        return container;
    }
    
    /**
     * Create gradient arrow for backpropagation
     */
    createGradientArrow(timeStep) {
        const geometry = new THREE.ConeGeometry(0.2, 1, 8);
        const material = new THREE.MeshLambertMaterial({ 
            color: this.colors.gradient,
            transparent: true,
            opacity: 0.8
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Point backward (negative Z direction)
        mesh.rotation.x = Math.PI;
        
        return mesh;
    }
    
    /**
     * Create connection line between objects
     */
    createConnectionLine(from, to) {
        const geometry = new THREE.BufferGeometry().setFromPoints([from, to]);
        const material = new THREE.LineBasicMaterial({ 
            color: this.colors.connection,
            transparent: true,
            opacity: 0.6
        });
        
        return new THREE.Line(geometry, material);
    }
    
    /**
     * Create text label
     */
    createTextLabel(text, color = 0xffffff) {
        // Create a canvas for the text
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 128;
        
        context.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
        context.font = '24px Arial';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        
        const lines = text.split('\n');
        lines.forEach((line, i) => {
            context.fillText(line, canvas.width / 2, canvas.height / 2 + (i - lines.length / 2 + 0.5) * 30);
        });
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.MeshBasicMaterial({ 
            map: texture, 
            transparent: true,
            alphaTest: 0.1
        });
        const geometry = new THREE.PlaneGeometry(2, 1);
        
        return new THREE.Mesh(geometry, material);
    }
    
    /**
     * Animation utilities
     */
    async animateObjectScale(object, targetScale, duration) {
        return new Promise(resolve => {
            const startScale = { ...object.scale };
            const startTime = Date.now();
            
            const animate = () => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = this.easeOutBack(progress);
                
                object.scale.x = startScale.x + (targetScale.x - startScale.x) * eased;
                object.scale.y = startScale.y + (targetScale.y - startScale.y) * eased;
                object.scale.z = startScale.z + (targetScale.z - startScale.z) * eased;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };
            
            animate();
        });
    }
    
    async animateDataFlow(timeStep, stepData) {
        // Animate input flowing into RNN cell
        const inputPosition = this.embeddingObjects[timeStep].position;
        const rnnPosition = this.rnnCells[timeStep].position;
        
        // Create flowing particle
        const particle = this.createFlowParticle();
        particle.position.copy(inputPosition);
        this.scene.add(particle);
        
        // Animate particle movement
        return new Promise(resolve => {
            const startTime = Date.now();
            const duration = 1000 / this.animationSpeed;
            
            const animate = () => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = this.easeInOutCubic(progress);
                
                particle.position.lerpVectors(inputPosition, rnnPosition, eased);
                particle.rotation.y += 0.1;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    this.scene.remove(particle);
                    
                    // Pulse the hidden state
                    this.pulseObject(this.hiddenStates[timeStep]);
                    resolve();
                }
            };
            
            animate();
        });
    }
    
    async animateGradientFlow(arrow, timeStep) {
        return new Promise(resolve => {
            const startTime = Date.now();
            const duration = 800 / this.animationSpeed;
            let opacity = 1;
            
            const animate = () => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                
                // Fade out the arrow
                opacity = 1 - progress;
                arrow.material.opacity = opacity;
                
                // Move arrow backward
                arrow.position.z -= 0.05;
                arrow.rotation.y += 0.05;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };
            
            animate();
        });
    }
    
    createFlowParticle() {
        const geometry = new THREE.SphereGeometry(0.1, 8, 8);
        const material = new THREE.MeshBasicMaterial({ 
            color: 0x00ff00,
            transparent: true,
            opacity: 0.8
        });
        
        return new THREE.Mesh(geometry, material);
    }
    
    pulseObject(object) {
        const originalScale = object.scale.x;
        const duration = 500;
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = elapsed / duration;
            const scale = originalScale + Math.sin(progress * Math.PI * 2) * 0.2;
            
            object.scale.setScalar(scale);
            
            if (progress < 2) {
                requestAnimationFrame(animate);
            } else {
                object.scale.setScalar(originalScale);
            }
        };
        
        animate();
    }
    
    // Easing functions
    easeOutBack(t) {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    }
    
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms / this.animationSpeed));
    }
    
    /**
     * Main animation loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls if available
        if (this.controls && this.controls.update) {
            this.controls.update();
        }
        
        // Update pulsing hidden states
        this.hiddenStates.forEach(state => {
            if (state.userData) {
                state.userData.pulsePhase += 0.02;
                const scale = state.userData.originalScale + Math.sin(state.userData.pulsePhase) * 0.1;
                state.scale.setScalar(scale);
            }
        });
        
        this.renderer.render(this.scene, this.camera);
    }
    
    /**
     * Handle window resize
     */
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    /**
     * Set animation speed
     */
    setAnimationSpeed(speed) {
        this.animationSpeed = speed;
    }
    
    /**
     * Get camera for external control
     */
    getCamera() {
        return this.camera;
    }
    
    /**
     * Get scene for external modifications
     */
    getScene() {
        return this.scene;
    }
}

// Export for use in other modules
window.Visualization3D = Visualization3D;
