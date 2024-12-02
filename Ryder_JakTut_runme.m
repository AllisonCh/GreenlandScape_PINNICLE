% Test case for PINNICLE at the Onset of Ryder Glacier (Tile 32_09)
% Gathering data into ISSM struct and running inversion to obtain basal friction coefficient C
% clear
steps=[0:4];

if any(steps == 0)
ISSMpath					= issmdir();
Path2data					= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/research/data/Greenland/';
Path2dataJAM				= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/Data/JAM/';
Path2dataGS					= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/Data/';


Now							= datetime;
Now.Format					= 'dd-MMM-uuuu_HH-mm-ss';
structSaveName				= strcat('./Models/Ryder_test_I', char(Now));
mdSaveName					= strcat('./Models/Ryder_test.', char(Now));
% Get tile boundaries
Tile						= '12_13'; % Ryder is 32_09
Region						= 'Helheim';

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

% Define desired model domain
Res							= 1.5e3;
Lx							= Tile_xmax - Tile_xmin;
Ly							= Tile_ymax - Tile_ymin;
nx							= Lx / Res;
ny							= Ly / Res;
end

FrictionLaw					= 'Weertman';
Friction_guess				= 4e3;
cost_fns					= [101 103];
cost_fns_coeffs				= [40, 1];

nsteps						= 1000;
maxiter_per_step			= 50;

%%
if any(steps==1) 
	disp('   Step 1: Mesh creation');
	md							= squaremesh(model,Lx, Ly, nx, ny);
	md.mesh.x					= md.mesh.x + Tile_xmin;
	md.mesh.y					= md.mesh.y + Tile_ymin;

	%Get observed fields on mesh nodes
	disp('   Loading BedMachine v5 data from NetCDF');
	ncdata						= strcat(Path2data,'BedMachineGreenland-v5.nc');
	x1							= double(ncread(ncdata,'x'))';
	y1							= double(flipud(ncread(ncdata,'y')));

	disp('   Loading velocity data from geotiff');
	[velx, R]					= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_x_filt_150m.tif'));
	vely						= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_y_filt_150m.tif'));
	vel							= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_filt_150m.tif'));
	velx						= flipud(velx);
	vely						= flipud(vely);
	vel							= flipud(vel);

	vx		= InterpFromGridToMesh(x1,y1,velx,md.mesh.x,md.mesh.y,0);
	vy		= InterpFromGridToMesh(x1,y1,vely,md.mesh.x,md.mesh.y,0);
	vel		= InterpFromGridToMesh(x1,y1,vel,md.mesh.x,md.mesh.y,0);

	save RydMesh md
end 

if any(steps==1) 
	disp('   Step 2: Parameterization');
	% md						= loadmodel('RydMesh');

	md						= setmask(md,'','');

	% Name and Coordinate system
	md.miscellaneous.name	= 'Ryder';
	md.mesh.epsg			= 3413;

	% Load rest of data
	disp('	Loading mask')
	Mask = readGeotiff(strcat(Path2dataGS,'GrIS_BM5_ice_sheet_mask_150m.tif'));
	Mask.y = flipud(Mask.y(:));
	mask_gris = flipud(Mask.z);


	disp('   Loading BedMachine v5 data from NetCDF');
	ncdata						= strcat(Path2data,'BedMachineGreenland-v5.nc');
	x1							= double(ncread(ncdata,'x'))';
	y1							= double(flipud(ncread(ncdata,'y')));
	bed							= single(rot90(ncread(ncdata, 'bed'))); %(topg)
	H							= single(rot90(ncread(ncdata, 'thickness'))); %(thk)
	bed(~mask_gris)				= NaN;
	H(~mask_gris)				= NaN;

	disp('   Loading GrIMP data from geotiff');
	h							= readgeoraster(strcat(Path2dataGS, 'GrIMP_30m_merged_filtered_150m.tif')); % surface elevation from a filtered, QGIS-resampled GeoTIFF of the original 30-m tiles, m
	h							= flipud(h);
	h(~mask_gris)				= NaN;
	
	disp('   Interpolating bedrock topography');
	md.geometry.base = InterpFromGridToMesh(x1,y1,bed,md.mesh.x,md.mesh.y,0);

	disp('	Interpolating surface elevation');
	md.geometry.surface			= InterpFromGridToMesh(x1, y1, h, md.mesh.x, md.mesh.y, 0);

	disp('   Loading velocity data from geotiff');
	[velx, R]					= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_x_filt_150m.tif'));
	vely						= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_y_filt_150m.tif'));
	vel							= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_filt_150m.tif'));
	velx						= flipud(velx);
	vely						= flipud(vely);
	vel							= flipud(vel);
	velx(~mask_gris)			= NaN;
	vely(~mask_gris)			= NaN;
	vel(~mask_gris)				= NaN;
	
	disp('   Interpolating velocities');
	md.inversion.vx_obs			= InterpFromGridToMesh(x1,y1,velx,md.mesh.x,md.mesh.y,0);
	md.inversion.vy_obs			= InterpFromGridToMesh(x1,y1,vely,md.mesh.x,md.mesh.y,0);
	md.inversion.vel_obs		= InterpFromGridToMesh(x1,y1,vel,md.mesh.x,md.mesh.y,0);
	md.initialization.vx		= md.inversion.vx_obs;
	md.initialization.vy		= md.inversion.vy_obs;
	md.initialization.vel		= md.inversion.vel_obs;

	% Get climate data from MAR
	disp('	Loading climate data from MAR')
	MAR							= load(strcat(Path2dataGS,'mar_311_Avg.mat'));
	temp						= MAR.TT_mean;
	smb							= MAR.SMB_mean * md.materials.rho_water ./md.materials.rho_ice * 365.25 / 1000; % convert mm WE day-1 to m yr-1

	disp('   Reconstruct thicknesses');
	md.geometry.thickness		= md.geometry.surface - md.geometry.base;
	pos0						= find(md.geometry.thickness<=10);
	md.geometry.thickness(pos0)	= 10;

	disp('	Reconstruct bed topography');
	md.geometry.base			= md.geometry.surface - md.geometry.thickness;

	disp('   Interpolating temperatures');
	md.initialization.temperature	= InterpFromGridToMesh(MAR.x(1,:),MAR.y(:,1),temp,md.mesh.x,md.mesh.y,0)+273.15; %convert to Kelvin

	disp('   Interpolating surface mass balance');
	md.smb.mass_balance			= InterpFromGridToMesh(MAR.x(1,:),MAR.y(:,1),smb,md.mesh.x,md.mesh.y,0);
end

if any(steps==2)

	% set rheology
	disp('   Construct ice rheological properties');
	md.materials.rheology_n		= 3*ones(md.mesh.numberofelements,1);
	md.materials.rheology_B		= paterson(md.initialization.temperature);
	md.damage.D					= zeros(md.mesh.numberofvertices,1);
	%Reduce viscosity along the shear margins
	% weakb						= ContourToMesh(md.mesh.elements,md.mesh.x,md.mesh.y,'WeakB.exp','node',2);
	% pos							= find(weakb);
	% md.materials.rheology_B(pos)= .3 * md.materials.rheology_B(pos);

	% Deal with boundary conditions
	disp('   Set other boundary conditions');
	% md							= SetMarineIceSheetBC(md,'./Front.exp');
	md								= SetIceSheetBC(md);
	md.basalforcings.floatingice_melting_rate = zeros(md.mesh.numberofvertices,1);
	md.basalforcings.groundedice_melting_rate = zeros(md.mesh.numberofvertices,1);

	% Set basal friction coefficient guess - frictionwaterlayer
	disp('   Construct basal friction parameters');
	disp('   Initial basal friction ');
	if strcmp(FrictionLaw,'waterlayer')
		md.friction.coefficient		= Friction_guess * ones(md.mesh.numberofvertices,1);
		md.friction.coefficient(find(md.mask.ocean_levelset<0.)) = 0.;
		md.friction.p				= ones(md.mesh.numberofelements,1);
		md.friction.q				= ones(md.mesh.numberofelements,1);
	elseif strcmp(FrictionLaw,'Weertman')
		%

		Wm							= 3; % m for Weertman friction law
		md.friction					= frictionweertman(); % Set friction law
		md.friction.m				= Wm .* ones(md.mesh.numberofelements, 1); % Set m exponent
		md.friction.C				= Friction_guess .*ones(md.mesh.numberofvertices, 1); % set reference friction coefficient
	end
	% md=parameterize(md,'Ryd.par');

	% save RydPar md
end 

if any(steps==3) 
	disp('   Step 3: Control method friction');
	% md=loadmodel('RydPar');

	md								= setflowequation(md,'SSA','all');

	%Control general
	md.inversion.iscontrol			= 1;
	md.inversion.nsteps				= nsteps; 
	md.inversion.step_threshold		= 0.99*ones(md.inversion.nsteps,1);
	md.inversion.maxiter_per_step	= maxiter_per_step*ones(md.inversion.nsteps,1);
	md.verbose						= verbose('solution',true,'control',true);

	%Cost functions
	md.inversion.cost_functions							= cost_fns;
	md.inversion.cost_functions_coefficients			= ones(md.mesh.numberofvertices,length(cost_fns));
	for ii = 1:length(cost_fns)
		md.inversion.cost_functions_coefficients(:,1)	= cost_fns_coeffs(ii);
	end

	% %Cost functions
	% md.inversion.cost_functions...
	% 	= [101 103 501]; % Specify the cost functions - these are summed to calculate final cost function; weights to each can be applied below
	% md.inversion.cost_functions_coefficients...
	% 	= zeros(md.mesh.numberofvertices, numel(md.inversion.cost_functions)); % initialize weights for cost functions
	% md.inversion.cost_functions_coefficients(:,1)...
	% 	= 1000; % weight for cost function 101
	% md.inversion.cost_functions_coefficients(:,2)...
	% 	= 180; % weight for cost function 103
	% md.inversion.cost_functions_coefficients(:,3)...
	% 	= 1.5e-8; % weight for cost function 501
	% pos							= find(md.mask.ice_levelset > 0); % Find where the ice mask is >0
	% md.inversion.cost_functions_coefficients(pos, 1:2)...
	% 	= 0; % Set coefficients to cost functions 101 and 103 to zero where the ice mask is >0

	%Controls
	if strcmp(FrictionLaw,'waterlayer')
		md.inverson.control_parameters	= {'FrictionCoefficient'};
	elseif strcmp(FrictionLaw,'Weertman')
		md.inversion.control_parameters	= {'FrictionC'};
	end
	
	md.inversion.gradient_scaling(1:md.inversion.nsteps)	= 30;
	md.inversion.min_parameters								= 1e-2 .* ones(md.mesh.numberofvertices,1);
	md.inversion.max_parameters								= 5e4 .* ones(md.mesh.numberofvertices,1);

	%Additional parameters
	md.stressbalance.restol			= 0.01;
	md.stressbalance.reltol			= 0.1;
	md.stressbalance.abstol			= NaN;

	%Go solve
	md.cluster						= generic('name',oshostname,'np',4);
	md								= solve(md,'Stressbalance');

	% save RydControl md

	% save in PINNICLE-friendly format
	warning off MATLAB:structOnObject
	md.friction						= struct(md.friction);
	warning on MATLAB:structOnObject
	if isfield(md.results.StressbalanceSolution,'FrictionC')
		md.friction.C_guess			= md.friction.C;
		md.friction.C				= md.results.StressbalanceSolution.FrictionC;
	else
		md.friction.C_guess			= md.friction.coefficient;
		md.friction.C				= md.results.StressbalanceSolution.FrictionCoefficient;
	end
end 



if any(steps==4) 

	disp('   Plotting')
	% md=loadmodel('RydControl');

	f1 = figure; plotmodel(md,'unit#all','km','axis#all','equal',...
		'data',md.inversion.vel_obs,'title','Observed velocity',...
		'data',md.results.StressbalanceSolution.Vel,'title','Modeled Velocity',...
		'colorbar#1','off','colorbartitle#2','(m/yr)',...
		'caxis#1',[0,150],...
		'data',md.geometry.base,'title','Base elevation',...
		'data',md.friction.C,...
		'title','Friction Coefficient',...
		'colorbartitle#3','(m)', 'figure', f1);

	f1 = figure; plotmodel(md,'unit#all','km','axis#all','image',...
		'data', md.initialization.vx, 'title', 'u','colorbartitle#1','m/yr',...
		'data', md.initialization.vy, 'title', 'v','colorbartitle#2','m/yr',...
		'data', md.geometry.surface, 'title', 'surface elev.','colorbartitle#3','m',...
		'data', md.friction.C_guess,'title','Friction Coefficient',...
		'colormap#1-2', cmocean('thermal'),'colormap#3',demcmap(md.geometry.surface), 'figure', f1)

end 


% save

if any(steps == 5)
	disp('	Saving')
	save(mdSaveName, 'md')
	saveasstruct(md, strcat(structSaveName, '.mat'));
end


%% Plotting - Weertman friction
% 
% 	f1 = figure; plotmodel(md,'unit#all','km','axis#all','equal',...
% 		'data',md.inversion.vel_obs,'title','Observed velocity',...
% 		'data',md.results.StressbalanceSolution.Vel,'title','Modeled Velocity',...
% 		'colorbar#1','off','colorbartitle#2','(m/yr)',...
% 		'caxis#1',[0,150],...
% 		'data',md.geometry.base,'title','Base elevation',...
% 		'data',md.results.StressbalanceSolution.FrictionC,...
% 		'title','Friction Coefficient',...
% 		'colorbartitle#3','(m)', 'figure', f1);
% 
% 	f1 = figure; plotmodel(md,'unit#all','km','axis#all','image',...
% 		'data', md.initialization.vx, 'title', 'u','colorbartitle#1','m/yr',...
% 		'data', md.initialization.vy, 'title', 'v','colorbartitle#2','m/yr',...
% 		'data', md.geometry.surface, 'title', 'surface elev.','colorbartitle#3','m',...
% 		'data', md.friction.C,'title','Friction Coefficient',...
% 		'colormap#1-2', cmocean('thermal'),'colormap#3',demcmap(md.geometry.surface), 'figure', f1)
% 
% 
% %% Plotting - water layer friction
% 
% 	f1 = figure; plotmodel(md,'unit#all','km','axis#all','equal',...
% 		'data',md.inversion.vel_obs,'title','Observed velocity',...
% 		'data',md.results.StressbalanceSolution.Vel,'title','Modeled Velocity',...
% 		'colorbar#1','off','colorbartitle#2','(m/yr)',...
% 		'caxis#1',[0,150],...
% 		'data',md.geometry.base,'title','Base elevation',...
% 		'data',md.results.StressbalanceSolution.FrictionCoefficient,...
% 		'title','Friction Coefficient',...
% 		'colorbartitle#3','(m)', 'figure', f1);
% 
% 	f1 = figure; plotmodel(md,'unit#all','km','axis#all','image',...
% 		'data', md.initialization.vx, 'title', 'u','colorbartitle#1','m/yr',...
% 		'data', md.initialization.vy, 'title', 'v','colorbartitle#2','m/yr',...
% 		'data', md.geometry.surface, 'title', 'surface elev.','colorbartitle#3','m',...
% 		'data', md.friction.coefficient,'title','Friction Coefficient',...
% 		'colormap#1-2', cmocean('thermal'),'colormap#3',demcmap(md.geometry.surface), 'figure', f1)
