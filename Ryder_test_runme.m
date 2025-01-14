% Test case for PINNICLE at the Onset of Ryder Glacier (Tile 32_09)
% Gathering data into ISSM struct and running inversion to obtain basal friction coefficient C
clear

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
flow_eq						= 'HO'; % 'SSA';
maxsteps					= 250;
maxiter						= 200;
cost_fns					= [101 103];
cost_fns_coeffs				= [40, 1];
% cost_fns					= [101 103 501];
% cost_fns_coeffs				= [1000 180 1.5e-8];
Friction_guess				= 3e3;

% Set constants (assume ice is at -5°C for now)
mu							= 2e8; %5 * 1.268020734014910e+08; % viscosity parameter
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

% Get climate data from MAR
	disp('	Loading climate data from MAR')
	MAR							= load(strcat(Path2dataGS,'mar_311_Avg.mat'));
	temp						= MAR.TT_mean;
	smb							= MAR.SMB_mean * md.materials.rho_water ./md.materials.rho_ice * 365.25 / 1000; % convert mm WE day-1 to m yr-1
	MAR.x						= MAR.x .* 1e3;
	MAR.y						= MAR.y .* 1e3;

	disp('   Interpolating temperatures');
	md.initialization.temperature	= InterpFromGridToMesh(MAR.x(1,:),MAR.y(:,1),temp,md.mesh.x,md.mesh.y,0)+273.15; %convert to Kelvin

	disp('   Interpolating surface mass balance');
	md.smb.mass_balance			= InterpFromGridToMesh(MAR.x(1,:),MAR.y(:,1),smb,md.mesh.x,md.mesh.y,0);

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
% md.materials.rheology_B		= mu .* ones(md.mesh.numberofvertices, 1); % ice rigidity
md.materials.rheology_B		= paterson(md.initialization.temperature);

%
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
md.friction.C				= Friction_guess .*ones(md.mesh.numberofvertices, 1); % set reference friction coefficient


% Extrude
if strcmp(flow_eq, 'HO')
	mds = extrude(md, 2, 1);
	% plotmodel(md, 'data', 'mesh')
	mds.basalforcings.groundedice_melting_rate = 0;
end

% Deal with boundary conditions:
disp('   Set Boundary conditions');
mds.stressbalance.spcvx		= NaN*ones(mds.mesh.numberofvertices,1); % x-axis velocity constraint (NaN means no constraint)
mds.stressbalance.spcvy		= NaN*ones(mds.mesh.numberofvertices,1); % y-axis velocity constraint (NaN means no constraint)
mds.stressbalance.spcvz		= NaN*ones(mds.mesh.numberofvertices,1); % z-axis velocity constraint (NaN means no constraint)
mds.stressbalance.referential= NaN*ones(mds.mesh.numberofvertices,6); % local referential - ?
mds.stressbalance.loadingforce = 0*ones(mds.mesh.numberofvertices,3); % ? set to zero
pos							= find((mds.mask.ice_levelset < 0) .* (mds.mesh.vertexonboundary)); % Find indices where ice mask is -1 and the vertex is on a boundary
mds.stressbalance.spcvx(pos) = mds.initialization.vx(pos); % Set the x-axis velocity constraint to the velocity on the boundary vertices
mds.stressbalance.spcvy(pos) = mds.initialization.vy(pos); % Set the y-axis velocity constraint to the velocity on the boundary vertices
mds.stressbalance.spcvz(pos) = 0;  % Set the z-axis velocity constraint to zero on the boundary vertices (no vertical component of velocity)
mds.stressbalance.spcvx_base = zeros(mds.mesh.numberofvertices,1);
mds.stressbalance.spcvy_base = zeros(mds.mesh.numberofvertices,1);

% No friction on PURELY ocean element (likely not necessary for these purposes)
pos_e						= find(min(mds.mask.ice_levelset(mds.mesh.elements), [], 2) < 0);  % Get the mask value at each vertex for each element, then take the min mask value in each row, then get the element indices where the mask is <0
flags						= ones(mds.mesh.numberofvertices, 1); % initialize flags array
flags(mds.mesh.elements(pos_e,:))...
							= 0; % set flags where elements have any vertices < 0 to 0
mds.friction.C(find(flags))	= 0.0; % Set friction coefficient to 0 where flags=1 (where no vertices of the element have ice)

mds							= setflowequation(mds, flow_eq, 'all'); % Set flow equation

mds.mask.ocean_levelset		= ones(mds.mesh.numberofvertices,1); % Initialize ocean mask

mds							= SetIceSheetBC(mds);

% Set up computation
mds.cluster					= generic('name',oshostname(),'np',4); % oshostname() to run on local computer, "'np',40" to request 40 cpus
mds.miscellaneous.name		= ['test']; % Give the run a name


%Control general
mds.inversion				= m1qn3inversion(mds.inversion); % Set the minimization algorithm to the M1QN3 algorithm (which is a thing outside of ISSM, even available as a Python module - see user guide 7.1.4.3)
mds.inversion.iscontrol		= 1; % make sure inversion flag is true
mds.verbose					= verbose('solution',false,'control',true); % specify how much output to print
mds.transient.amr_frequency...
							= 0; % Do not run with AMR (adaptive mesh refinement - requires extra installation of NewPZ)

%Cost functions
mds.inversion.cost_functions							= cost_fns;
mds.inversion.cost_functions_coefficients			= ones(mds.mesh.numberofvertices,length(cost_fns));
for ii = 1:length(cost_fns)
	mds.inversion.cost_functions_coefficients(:,1)	= cost_fns_coeffs(ii);
end
% md.inversion.cost_functions...
% 							= [101 103 501]; % Specify the cost functions - these are summed to calculate final cost function; weights to each can be applied below
% md.inversion.cost_functions_coefficients...
% 							= zeros(md.mesh.numberofvertices, numel(md.inversion.cost_functions)); % initialize weights for cost functions
% md.inversion.cost_functions_coefficients(:,1)...
% 							= 1000; % weight for cost function 101
% md.inversion.cost_functions_coefficients(:,2)...
% 							= 180; % weight for cost function 103
% md.inversion.cost_functions_coefficients(:,3)...
% 							= 1.5e-8; % weight for cost function 501
% pos							= find(md.mask.ice_levelset > 0); % Find where the ice mask is >0
% md.inversion.cost_functions_coefficients(pos, 1:2)...
% 							= 0; % Set coefficients to cost functions 101 and 103 to zero where the ice mask is >0

%Controls
mds.inversion.control_parameters...
							= {'FrictionC'}; % Set parameter to be inferred ('FrictionC' or 'FrictionCoefficient')
mds.inversion.maxsteps		= maxsteps; % Set max number of iterations (gradient computation - M1QN3 specific)
mds.inversion.maxiter		= maxiter; % Set max number of Function evaluations (forward run - M1QN3 specific)
mds.inversion.min_parameters	= 0.01 .* ones(mds.mesh.numberofvertices,1); % minimum value for the inferred parameter (FrictionC)
mds.inversion.max_parameters	= 5e4 .* ones(mds.mesh.numberofvertices,1); % maximum value for the inferred parameter (FrictionC)
mds.inversion.control_scaling_factors...
							= 1; % ? not in user guide
mds.inversion.dxmin			= 1e-6; % Convergence criterion: two points less than dxmin from eachother (sup-norm) are considered identical (M1QN3 specific)
%Additional parameters
mds.stressbalance.restol		= 1e-6; % stress balance tolerance to avoid accumulation of numerical residuals between consecutive samples (mechanical equilibrium residue convergence criterion)
mds.stressbalance.reltol		= 0.01; % velocity relative convergence criterion
mds.stressbalance.abstol		= NaN; % velocity absolute convergence criterion

mds.toolkits.DefaultAnalysis	= bcgslbjacobioptions(); % toolkits are PETSc options for each solution (PETSc is an external package used in ISSM)
% md.verbose = verbose('solution',true,'control',true, 'convergence', true);
mds							= solve(mds,'Stressbalance'); % Solve the model


% save in PINNICLE-friendly format
warning off MATLAB:structOnObject
md.friction						= struct(md.friction);
warning on MATLAB:structOnObject
md.friction.C_guess				= md.friction.C;
if strcmp(flow_eq, 'HO')
	md.friction.C				= InterpFromModel3dToMesh2d(mds,mds.results.StressbalanceSolution.FrictionC, md.mesh.x, md.mesh.y,0,Friction_guess);
else
	md.friction.C				= mds.results.StressbalanceSolution.FrictionC;
end


f4 = figure;
plotmodel(md, 'data', md.friction.C, 'figure', f4)
% 
% save(mdSaveName, 'md')
% saveasstruct(md, strcat(structSaveName, '.mat'));
%%
disp('   Plotting')
	% md=loadmodel('RydControl');

	f1 = figure; plotmodel(mds,'unit#all','km','axis#all','equal',...
		'data',mds.inversion.vel_obs,'title','Observed velocity',...
		'data',mds.results.StressbalanceSolution.Vel,'title','Modeled Velocity',...
		'colorbar#1','off','colorbartitle#2','(m/yr)',...
		'caxis#1',[0,150],...
		'data',mds.geometry.base,'title','Base elevation',...
		'data',mds.results.StressbalanceSolution.FrictionC,...
		'title','Friction Coefficient',...
		'colorbartitle#3','(m)', 'figure', f1);

	f1 = figure; plotmodel(mds,'unit#all','km','axis#all','image',...
		'data', mds.initialization.vx, 'title', 'u','colorbartitle#1','m/yr',...
		'data', mds.initialization.vy, 'title', 'v','colorbartitle#2','m/yr',...
		'data', mds.geometry.surface, 'title', 'surface elev.','colorbartitle#3','m',...
		'data', mds.results.StressbalanceSolution.FrictionC,'title','Friction Coefficient',...
		'colormap#1-2', cmocean('thermal'),'colormap#3',demcmap(mds.geometry.surface), 'figure', f1)