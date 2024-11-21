% Test case for PINNICLE at the Onset of Ryder Glacier (Tile 32_09)
% Gathering data into ISSM struct and running inversion to obtain basal friction coefficient C
% clear

ISSMpath					= issmdir();
Path2data					= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/research/data/Greenland/';
Path2dataJAM				= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/Data/JAM/';
Path2dataGS					= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/Data/';

Now							= datetime;
Now.Format					= 'dd-MMM-uuuu_HH-mm-ss';
structSaveName				= strcat('./Models/Ryder_test_I', char(Now));
mdSaveName					= strcat('./Models/Ryder_test.', char(Now));
% Get tile boundaries
Tile						= '32_09';
Region						= 'Ryder';

load(strcat(Path2dataGS,'GreenlandScape_Tiles.mat'))
for ii = 1:length(Tiles)
	Tile_names{ii}			= Tiles(ii).name;
end

Tile_idx					= find(strcmp(Tile_names, Tile));
Tile_xmin					= Tiles(Tile_idx).X(1) * 1e3;
Tile_xmax					= Tiles(Tile_idx).X(2) * 1e3;
Tile_ymin					= Tiles(Tile_idx).Y(1) * 1e3;
Tile_ymax					= Tiles(Tile_idx).Y(2) * 1e3;
Tile_XY_pos					= [Tile_xmin, Tile_ymin; Tile_xmax, Tile_ymin; Tile_xmax, Tile_ymax; Tile_xmin, Tile_ymax; Tile_xmin, Tile_ymin];

% Write .exp file
domainFilename				= strcat(Region,'_',Tile,'.exp');
if ~exist(domainFilename, 'file')
	fileID					= fopen(domainFilename,'w');
	fprintf(fileID, '## Name:domainoutline\n');
	fprintf(fileID, '## Icon:0\n');
	fprintf(fileID, '# Points Count Value\n');
	fprintf(fileID, '5 1.\n');
	fprintf(fileID, '# X pos Y pos\n');
	fprintf(fileID, '%d %d\n',Tile_XY_pos');
	fclose(fileID);
end

% Define desired model domain
Res							= 1.5e3;
Lx							= Tile_xmax - Tile_xmin;
Ly							= Tile_ymax - Tile_ymin;
nx							= Lx / Res;
ny							= Ly / Res;

% Set Model choices
flow_eq						= 'SSA';
maxsteps					= 250;
maxiter						= 200;
cost_fns					= [101 103 501];
cost_fns_coeffs				= [1000 180 1.5e-8];

% Set constants (assume ice is at -5°C for now)
mu							= 5 * 1.268020734014910e+08; % viscosity parameter
Gn							= 3; % Glen's exponent
Wm							= 3; % m for Weertman friction law


%% Build model
% Create mesh
disp('	Creating 2D mesh');
% md							=triangle(model,domainFilename,Res);
md							= squaremesh(model,Lx, Ly, nx, ny);
md.mesh.x					= md.mesh.x + Tile_xmin;
md.mesh.y					= md.mesh.y + Tile_ymin;

% md = parameterize(md,'./Greenlandscape.par');

% Get geometry 
disp('   Loading BedMachine v5 data from NetCDF');
ncdata						= strcat(Path2data,'BedMachineGreenland-v5.nc');
x1							= double(ncread(ncdata,'x'))';
y1							= double(flipud(ncread(ncdata,'y')));
bed							= single(rot90(ncread(ncdata, 'bed')));
H							= single(rot90(ncread(ncdata, 'thickness')));

disp('   Loading GrIMP data from geotiff');
h							= readgeoraster(strcat(Path2dataGS, 'GrIMP_30m_merged_filtered_150m.tif')); % surface elevation from a filtered, QGIS-resampled GeoTIFF of the original 30-m tiles, m
h							= flipud(h);

disp('   Interpolating surface and bedrock');
md.geometry.bed				= InterpFromGridToMesh(x1, y1, bed, md.mesh.x, md.mesh.y, 0);
md.geometry.base			= InterpFromGridToMesh(x1, y1, bed, md.mesh.x, md.mesh.y, 0);
md.geometry.surface			= InterpFromGridToMesh(x1, y1, h, md.mesh.x, md.mesh.y, 0);

% Read in velocities
disp('   Loading velocity data from geotiff');
[velx, R]					= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_x_filt_150m.tif'));
vely						= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_y_filt_150m.tif'));
vel							= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_filt_150m.tif'));
velx						= flipud(velx);
vely						= flipud(vely);
vel							= flipud(vel);

disp('   Interpolating velocities ');
md.inversion.vx_obs			= InterpFromGridToMesh(x1, y1, velx, md.mesh.x, md.mesh.y, 0); % set observed x-velocity for inversion
md.inversion.vy_obs			= InterpFromGridToMesh(x1, y1, vely, md.mesh.x, md.mesh.y, 0); % set observed y-velocity for inversion
md.inversion.vel_obs		= InterpFromGridToMesh(x1, y1, vel, md.mesh.x, md.mesh.y, 0); % set observed velocity magnitude for inversion
md.initialization.vx		= md.inversion.vx_obs; % set initial/observed x-velocity
md.initialization.vy		= md.inversion.vy_obs; % set initial/observed x-velocity
md.initialization.vz		= zeros(md.mesh.numberofvertices, 1); % set vertical component of velocity to zero
md.initialization.vel		= md.inversion.vel_obs; % set initial/observed velocity magnitude


% Set masks (md.mask.ice_levelset < 0 where ice present; > 0 where no ice, = 0 at ice front)
disp('   Setting ice mask');
icemask						= readgeoraster(strcat(Path2dataGS, 'GrIS_BM5_ice_sheet_mask_150m.tif'));
icemask						= InterpFromGridToMesh(x1, y1, icemask, md.mesh.x, md.mesh.y, 0);
icemask_md					= -1 .* ones(size(md.mesh.x)); % initialize ice mask
icemask_md(icemask < 0)		= 1; % set model icemask to 1 (no ice) where appropriate
md.mask.ice_levelset		= icemask_md; % load ice mask into model
% set surface so that minimum thickness is 10, even where ice mask is positive
pos							= find(md.mask.ice_levelset > 0); % Find where the ice mask is positive (no ice present)
md.geometry.surface(pos)	= md.geometry.base(pos) + 10; %Minimum thickness; set the surface equal to the bed height + 10 where ice mask is positive
md.geometry.thickness		= md.geometry.surface - md.geometry.bed; % recompute thickness
pos							= find(md.geometry.thickness <= 10); % find where thickness is < 10
md.geometry.surface(pos)	= md.geometry.base(pos) + 10; %Minimum thickness; set the surface equal to the bet height + 10 where thickness is < 10
md.geometry.thickness		= md.geometry.surface - md.geometry.bed; % recompute thickness
md.masstransport.min_thickness...
							= 10; % set the min thickness in the model
disp('	Adjusting ice mask');
%Tricky part here: we want to offset the mask by one element so that we don't end up with a cliff at the transition
pos							= find(max(md.mask.ice_levelset(md.mesh.elements),[],2)>0); % Get the mask value at each vertex for each element, then take the max mask value in each row, then get the element indices where the mask is >0
md.mask.ice_levelset(md.mesh.elements(pos,:))...
							= 1; % sets mask value of any vertices connected to an element with any vertex with a mask value of 1 to 1 also
% For the region where surface is NaN, set thickness to small value (consistency requires >0)
pos							= find((md.mask.ice_levelset<0).*(md.geometry.surface<0)); % find indices of vertices where the mask is -1 and the surface is < 0
md.mask.ice_levelset(pos)	= 1; % set the mask at these vertices to 1
pos							= find((md.mask.ice_levelset<0).*(isnan(md.geometry.surface))); % find indices of vertices where the mask is -1 and the surface is NaN
md.mask.ice_levelset(pos)	= 1; % set the mask at these vertices to 1

disp('      -- reconstruct thickness');
md.geometry.thickness		= md.geometry.surface - md.geometry.base;

% Done with input data
clear ncdata x1 y1 bed H h velx vely vel

% set rheology
disp('   Creating flow law parameters (assume ice is at -5°C for now)');
md.materials.rheology_n		= Gn .* ones(md.mesh.numberofelements, 1); % Glen's exponent
md.materials.rheology_B		= mu .* ones(md.mesh.numberofvertices, 1); % ice rigidity

% Extrude
if strcmp(flow_eq, 'HO')
	md = extrude(md, 2, 1);
	% plotmodel(md, 'data', 'mesh')
	md.basalforcings.groundedice_melting_rate = 0;
end

% Deal with boundary conditions:
disp('   Set Boundary conditions');
md.stressbalance.spcvx		= NaN*ones(md.mesh.numberofvertices,1); % x-axis velocity constraint (NaN means no constraint)
md.stressbalance.spcvy		= NaN*ones(md.mesh.numberofvertices,1); % y-axis velocity constraint (NaN means no constraint)
md.stressbalance.spcvz		= NaN*ones(md.mesh.numberofvertices,1); % z-axis velocity constraint (NaN means no constraint)
md.stressbalance.referential= NaN*ones(md.mesh.numberofvertices,6); % local referential - ?
md.stressbalance.loadingforce = 0*ones(md.mesh.numberofvertices,3); % ? set to zero
pos							= find((md.mask.ice_levelset < 0) .* (md.mesh.vertexonboundary)); % Find indices where ice mask is -1 and the vertex is on a boundary
md.stressbalance.spcvx(pos) = md.initialization.vx(pos); % Set the x-axis velocity constraint to the velocity on the boundary vertices
md.stressbalance.spcvy(pos) = md.initialization.vy(pos); % Set the y-axis velocity constraint to the velocity on the boundary vertices
md.stressbalance.spcvz(pos) = 0;  % Set the z-axis velocity constraint to zero on the boundary vertices (no vertical component of velocity)
md.stressbalance.spcvx_base = zeros(md.mesh.numberofvertices,1);
md.stressbalance.spcvy_base = zeros(md.mesh.numberofvertices,1);
%%
% Set basal friction coefficient guess - frictionwaterlayer
disp('   Construct basal friction parameters');
% md.friction.coefficient		= 1e2 *ones(md.mesh.numberofvertices,1);
% md.friction.coefficient(find(md.mask.ocean_levelset<0.)) = 0.;
% md.friction.p				= ones(md.mesh.numberofelements,1);
% md.friction.q				= ones(md.mesh.numberofelements,1);

% Set basal friction coefficient guess
disp('   Initial basal friction ');
md.friction					= frictionweertman(); % Set friction law
md.friction.m				= Wm .* ones(md.mesh.numberofelements, 1); % Set m exponent
md.friction.C				= 100 .*ones(md.mesh.numberofvertices, 1); % set reference friction coefficient

% No friction on PURELY ocean element (likely not necessary for these purposes)
pos_e						= find(min(md.mask.ice_levelset(md.mesh.elements), [], 2) < 0);  % Get the mask value at each vertex for each element, then take the min mask value in each row, then get the element indices where the mask is <0
flags						= ones(md.mesh.numberofvertices, 1); % initialize flags array
flags(md.mesh.elements(pos_e,:))...
							= 0; % set flags where elements have any vertices < 0 to 0
md.friction.C(find(flags))	= 0.0; % Set friction coefficient to 0 where flags=1 (where no vertices of the element have ice)

md							= setflowequation(md, flow_eq, 'all'); % Set flow equation

md.mask.ocean_levelset		= ones(md.mesh.numberofvertices,1); % Initialize ocean mask

md							= SetIceSheetBC(md);

% Set up computation
md.cluster					= generic('name',oshostname(),'np',2); % oshostname() to run on local computer, "'np',40" to request 40 cpus
md.miscellaneous.name		= ['test']; % Give the run a name


%Control general
md.inversion				= m1qn3inversion(md.inversion); % Set the minimization algorithm to the M1QN3 algorithm (which is a thing outside of ISSM, even available as a Python module - see user guide 7.1.4.3)
md.inversion.iscontrol		= 1; % make sure inversion flag is true
md.verbose					= verbose('solution',false,'control',true); % specify how much output to print
md.transient.amr_frequency...
							= 0; % Do not run with AMR (adaptive mesh refinement - requires extra installation of NewPZ)

%Cost functions
md.inversion.cost_functions...
							= [101 103 501]; % Specify the cost functions - these are summed to calculate final cost function; weights to each can be applied below
md.inversion.cost_functions_coefficients...
							= zeros(md.mesh.numberofvertices, numel(md.inversion.cost_functions)); % initialize weights for cost functions
md.inversion.cost_functions_coefficients(:,1)...
							= 1000; % weight for cost function 101
md.inversion.cost_functions_coefficients(:,2)...
							= 180; % weight for cost function 103
md.inversion.cost_functions_coefficients(:,3)...
							= 1.5e-8; % weight for cost function 501
pos							= find(md.mask.ice_levelset > 0); % Find where the ice mask is >0
md.inversion.cost_functions_coefficients(pos, 1:2)...
							= 0; % Set coefficients to cost functions 101 and 103 to zero where the ice mask is >0

%Controls
md.inversion.control_parameters...
							= {'FrictionC'}; % Set parameter to be inferred ('FrictionC' or 'FrictionCoefficient')
md.inversion.maxsteps		= maxsteps; % Set max number of iterations (gradient computation - M1QN3 specific)
md.inversion.maxiter		= maxiter; % Set max number of Function evaluations (forward run - M1QN3 specific)
md.inversion.min_parameters	= 0.01 .* ones(md.mesh.numberofvertices,1); % minimum value for the inferred parameter (FrictionC)
md.inversion.max_parameters	= 5e4 .* ones(md.mesh.numberofvertices,1); % maximum value for the inferred parameter (FrictionC)
md.inversion.control_scaling_factors...
							= 1; % ? not in user guide
md.inversion.dxmin			= 1e-6; % Convergence criterion: two points less than dxmin from eachother (sup-norm) are considered identical (M1QN3 specific)
%Additional parameters
md.stressbalance.restol		= 0.01; % stress balance tolerance to avoid accumulation of numerical residuals between consecutive samples (mechanical equilibrium residue convergence criterion)
md.stressbalance.reltol		= 0.1; % velocity relative convergence criterion
md.stressbalance.abstol		= NaN; % velocity absolute convergence criterion

md.toolkits.DefaultAnalysis	= bcgslbjacobioptions(); % toolkits are PETSc options for each solution (PETSc is an external package used in ISSM)
% md.verbose = verbose('solution',true,'control',true, 'convergence', true);
md							= solve(md,'Stressbalance'); % Solve the model

% md.friction.C				= md.results.StressbalanceSolution.FrictionC;
% f4 = figure;
% plotmodel(md, 'data', md.friction.C, 'figure', f4)
% 
% save(mdSaveName, 'md')
% saveasstruct(md, strcat(structSaveName, '.mat'));
%%
disp('   Plotting')
	% md=loadmodel('RydControl');

	f1 = figure; plotmodel(md,'unit#all','km','axis#all','equal',...
		'data',md.inversion.vel_obs,'title','Observed velocity',...
		'data',md.results.StressbalanceSolution.Vel,'title','Modeled Velocity',...
		'colorbar#1','off','colorbartitle#2','(m/yr)',...
		'caxis#1',[0,150],...
		'data',md.geometry.base,'title','Base elevation',...
		'data',md.results.StressbalanceSolution.FrictionC,...
		'title','Friction Coefficient',...
		'colorbartitle#3','(m)', 'figure', f1);

	f1 = figure; plotmodel(md,'unit#all','km','axis#all','image',...
		'data', md.initialization.vx, 'title', 'u','colorbartitle#1','m/yr',...
		'data', md.initialization.vy, 'title', 'v','colorbartitle#2','m/yr',...
		'data', md.geometry.surface, 'title', 'surface elev.','colorbartitle#3','m',...
		'data', md.results.StressbalanceSolution.FrictionC,'title','Friction Coefficient',...
		'colormap#1-2', cmocean('thermal'),'colormap#3',demcmap(md.geometry.surface), 'figure', f1)