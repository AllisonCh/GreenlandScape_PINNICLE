% Test case for PINNICLE at the Onset of Ryder Glacier (Tile 32_09)
% Gathering data into ISSM struct and running inversion to obtain basal friction coefficient C
% clear
steps=[0:5];

if any(steps == 0)
ISSMpath					= issmdir();
Path2data					= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/research/data/Greenland/';
Path2dataJAM				= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/Data/JAM/';
Path2dataGS					= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/Data/';

% Get tile boundaries
Tile						= '17_11'; % Ryder is 32_09; East of Jackobsavn (EGIG) is 17_11
Region						= 'UpperJakobshavn';

% Specify save names
RunNum						= '1'; % CHANGE THIS EACH TIME!!!
Now							= datetime;
Now.Format					= 'uuuu-MMM-dd'; % 'dd-MMM-uuuu_HH-mm-ss';
structSaveName				= strcat('./Models/',Region,'_issm', char(Now),'_',RunNum);
mdSaveName					= strcat('./Models/',Region,'.', char(Now),'_',RunNum);



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
Res							= 5e2;
Lx							= Tile_xmax - Tile_xmin;
Ly							= Tile_ymax - Tile_ymin;
nx							= Lx / Res;
ny							= Ly / Res;
end

flow_eq						= 'HO';

FrictionLaw					= 'Weertman';
Friction_guess				= 2e3;
cost_fns					= [101 103];
cost_fns_coeffs				= [40, 1];

Gn							= 3;

nsteps						= 100;
maxiter_per_step			= 20;

%%
if any(steps==1) 
	disp('   Step 1: Mesh creation');
	md							= squaremesh(model,Lx, Ly, nx, ny);
	md.mesh.x					= md.mesh.x + Tile_xmin;
	md.mesh.y					= md.mesh.y + Tile_ymin;

	% save(strcat(Region,'Mesh'),'md')
end 


if any(steps==1) 
	disp('   Step 2: Parameterization');
	% md							= loadmodel(strcat(Region,'Mesh'));

	md							= setmask(md,'','');

	% Name and Coordinate system
	md.miscellaneous.name		= Region;
	md.mesh.epsg				= 3413;

	% Load rest of data
	disp('	Loading mask')
	Mask						= readGeotiff(strcat(Path2dataGS,'GrIS_BM5_ice_sheet_mask_150m.tif'));
	Mask.y						= flipud(Mask.y(:));
	mask_gris					= flipud(Mask.z);


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
	MAR.x						= MAR.x .* 1e3;
	MAR.y						= MAR.y .* 1e3;

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
	md.materials.rheology_n		= Gn*ones(md.mesh.numberofelements,1);
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

	% Extrude if necessary
	if strcmp(flow_eq, 'HO')
		mds = extrude(md, 2, 1);
		% plotmodel(md, 'data', 'mesh')
		mds.basalforcings.groundedice_melting_rate = 0;


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
	else
		mds = md;
	end
	% Set basal friction coefficient guess - frictionwaterlayer
	disp('   Construct basal friction parameters');
	disp('   Initial basal friction ');

	% md=parameterize(md,'Ryd.par');

	% save(strcat(Region,'Par'),'mds')
end 

if any(steps==3) 
	disp('   Step 3: Control method friction');
	% mds=loadmodel(strcat(Region,'Par'));

	mds								= setflowequation(mds,flow_eq,'all');

	%Control general
	mds.inversion.iscontrol			= 1;
	mds.inversion.nsteps				= nsteps; 
	mds.inversion.step_threshold		= 0.99*ones(mds.inversion.nsteps,1);
	mds.inversion.maxiter_per_step	= maxiter_per_step*ones(mds.inversion.nsteps,1);
	mds.verbose						= verbose('solution',true,'control',true);

	%Cost functions
	mds.inversion.cost_functions							= cost_fns;
	mds.inversion.cost_functions_coefficients			= ones(mds.mesh.numberofvertices,length(cost_fns));
	for ii = 1:length(cost_fns)
		mds.inversion.cost_functions_coefficients(:,1)	= cost_fns_coeffs(ii);
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
		mds.inverson.control_parameters	= {'FrictionCoefficient'};
	elseif strcmp(FrictionLaw,'Weertman')
		mds.inversion.control_parameters	= {'FrictionC'};
	end
	
	mds.inversion.gradient_scaling(1:mds.inversion.nsteps)	= 30;
	mds.inversion.min_parameters								= 1e-2 .* ones(mds.mesh.numberofvertices,1);
	mds.inversion.max_parameters								= 5e4 .* ones(mds.mesh.numberofvertices,1);

	%Additional parameters
	mds.stressbalance.restol			= 1e-5;
	mds.stressbalance.reltol			= 0.1;
	mds.stressbalance.abstol			= NaN;

	%Go solve
	mds.cluster						= generic('name',oshostname,'np',4);
	mds								= solve(mds,'Stressbalance');

	% save(strcat(Region,'Control'), 'mds')

	% save in PINNICLE-friendly format
	warning off MATLAB:structOnObject
	md.friction						= struct(md.friction);
	warning on MATLAB:structOnObject

if isfield(mds.results.StressbalanceSolution,'FrictionC')
	md.friction.C_guess			= md.friction.C;
	if strcmp(flow_eq, 'HO')
		md.friction.C				= InterpFromModel3dToMesh2d(mds,mds.results.StressbalanceSolution.FrictionC, md.mesh.x, md.mesh.y,0,Friction_guess);
	else
		md.friction.C				= mds.results.StressbalanceSolution.FrictionC;
	end
else
	md.friction.C_guess			= md.friction.coefficient;
	md.friction.C				= mds.results.StressbalanceSolution.FrictionCoefficient;
end

end 

% save

if any(steps == 4)
	disp('	Saving')
	save(mdSaveName, 'mds')
	saveasstruct(md, strcat(structSaveName, '.mat'));
end

% if any(steps==5) 

	disp('   Plotting')
	% mds=loadmodel(mdSaveName);
	% 
	f1 = figure; plotmodel(mds,'unit#all','km','axis#all','equal',...
		'data',mds.inversion.vel_obs,'layer',1,'title','Observed velocity',...
		'data',mds.results.StressbalanceSolution.Vel,'layer',1,'title','Modeled Velocity',...
		'colorbar#1','off','colorbartitle#2','(m/yr)',...
		'caxis#1',[0,150],...
		'data',mds.geometry.base,'layer',2,'title','Base elevation',...
		'data',mds.friction.C,'layer',2,...
		'title','Friction Coefficient',...
		'colorbartitle#3','(m)', 'figure', f1);

	f1 = figure; plotmodel(md,'unit#all','km','axis#all','image',...
		'data', md.inversion.vx_obs, 'title', 'u','colorbartitle#1','m/yr',...
		'data', md.inversion.vy_obs, 'title', 'v','colorbartitle#2','m/yr',...
		'data', md.geometry.surface, 'title', 'surface elev.','colorbartitle#3','m',...
		'data', md.friction.C,'title','Friction Coefficient',...
		'data', md.geometry.thickness, 'title', 'thickness',...
		'data', md.smb.mass_balance, 'title', 'mass balance',...
		'data', md.balancethickness.thickening_rate, 'title','thickening rate',...
		'colormap#1-2', cmocean('thermal'),'colormap#3',demcmap(md.geometry.surface), 'figure', f1)

% end 





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


%% Get xyz radar data

OIB                         = load('/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/Data/JAM/xyz_all.mat');

% Get OIB flight lines within tile for minimization problem
OIBf						= struct('campaign',[],'num_segment',[],'num_frame',[],'gd_segment',[],'x',[],'y',[],'elev_surf',[],'elev_bed',[]); % Set up struct for data within bounding box
OIBf.gd_campaign			= []; % Set up variable to record campaigns with points in roi

for ii = 1:OIB.num_campaign % Loop through campaigns
    OIBf.campaign{ii}		= OIB.campaign{ii}; % Campaign name
    OIBf.num_segment(ii)	= OIB.num_segment(ii); % Number of days from each campaign (e.g. campaign 1993_Greenland_P3 has 10 segments (days))
    OIBf.num_frame{ii}		= OIB.num_frame{ii}; % Number of frames from each day of each campaign (e.g. segment 19930623_01 from 1993_Greenland_P3 has 32 frames)
    OIBf.segment{ii}		= OIB.segment{ii}; % Days of each campaign (e.g. 1x10 cell for campaign 1993_Greenland_P3 listing the first frame of each day, e.g. 19930623_01)
    OIBf.gd_segment{ii}		= []; % Set up variable to record segments with points in roi
    OIBf.x{ii}				= [];
    OIBf.y{ii}				= [];

	for j = 1:OIB.num_segment(ii)      % Loop through each segment from each campaign
		% Get indices of segment points within tile
        n					= find(OIB.x{ii}{j} >= (Tile_xmin) & OIB.x{ii}{j}...
            <= (Tile_xmax) & OIB.y{ii}{j} >= (Tile_ymin) & OIB.y{ii}{j} <= (Tile_ymax));

        if ~isempty(n) && length(n) > 2 % If there are more than 2 points per segment, combine with other segments
        OIBf.elev_bed{ii}{j}	= OIB.elev_bed{ii}{j}(n); % bed elevation
        OIBf.elev_surf{ii}{j}	= OIB.elev_surf{ii}{j}(n); % elevation of ice surface
        OIBf.x{ii}{j}			= OIB.x{ii}{j}(n); % PS x
        OIBf.y{ii}{j}			= OIB.y{ii}{j}(n); % PX y 
        OIBf.gd_segment{ii}		= [OIBf.gd_segment{ii}, j]; % Record only segments with points in roi
        end
        clear n
	end

    % Only keep segments with points in tile
	if ~isempty(OIBf.x{ii})
    OIBf.elev_bed{ii}		= OIBf.elev_bed{ii}(OIBf.gd_segment{ii});
    OIBf.elev_surf{ii}		= OIBf.elev_surf{ii}(OIBf.gd_segment{ii});
    OIBf.x{ii}				= OIBf.x{ii}(OIBf.gd_segment{ii});
    OIBf.y{ii}				= OIBf.y{ii}(OIBf.gd_segment{ii});
    OIBf.segment{ii}		= OIBf.segment{ii}(OIBf.gd_segment{ii});
    OIBf.num_frame{ii}		= OIBf.num_frame{ii}(OIBf.gd_segment{ii});
    OIBf.gd_campaign		= [OIBf.gd_campaign, ii];
	end
end

% Only keep campaigns with points in tile
OIBf.elev_bed               = OIBf.elev_bed(OIBf.gd_campaign);
OIBf.elev_surf              = OIBf.elev_surf(OIBf.gd_campaign);
OIBf.x                      = OIBf.x(OIBf.gd_campaign);
OIBf.y                      = OIBf.y(OIBf.gd_campaign);
OIBf.segment                = OIBf.segment(OIBf.gd_campaign);
OIBf.num_frame              = OIBf.num_frame(OIBf.gd_campaign);
OIBf.num_segment            = OIBf.num_segment(OIBf.gd_campaign);
OIBf.campaign               = OIBf.campaign(OIBf.gd_campaign);
OIBf.gd_segment             = OIBf.gd_segment(OIBf.gd_campaign);

% Aggregate observations from all years/campaigns/segments
OIBc                        = struct('xq', [], 'yq', [], 'thickq', [], 'campaign', [], 'segment', [], 'x', [], 'y', [], 'thick', []);

for ii = 1:length(OIBf.campaign) % Loop through campaigns
    for j = 1:length(OIBf.segment{ii}) % Loop through segments
    OIBc.xq                 = [OIBc.xq; OIBf.x{ii}{j}(:)];
    OIBc.yq                 = [OIBc.yq; OIBf.y{ii}{j}(:)];
    OIBc.thickq             = [OIBc.thickq; OIBf.elev_surf{ii}{j}(:) - OIBf.elev_bed{ii}{j}(:)];
    OIBc.campaign           = [OIBc.campaign; OIBf.campaign(ii)];
    OIBc.segment            = [OIBc.segment; OIBf.segment{ii}{j}];
    end
end

% Filter out infinite values
if any(OIBc.thickq == Inf)
OIBc.xq(OIBc.thickq == Inf)	= [];
OIBc.yq(OIBc.thickq == Inf) = [];
OIBc.thickq(OIBc.thickq == Inf)	= [];
end


Loc							= isnan(OIBc.thickq);
OIBc.xq(Loc)						= [];
OIBc.yq(Loc)						= [];
OIBc.thickq(Loc)					= [];

hnx							= x1 <= Tile_xmax + 150 & x1 >= Tile_xmin - 150;
hny							= y1 <= Tile_ymax + 150 & y1 >= Tile_ymin - 150;

x_tile						= x1(hnx);
y_tile						= y1(hny);
h_tile						= h(hny, hnx);
num_x_tile					= length(x_tile);
num_y_tile					= length(y_tile);

[x_grd, y_grd]				= meshgrid(x_tile, y_tile);

% Resample observations at BM5 grid resolution, taking the median of all points within a grid cell
kx							= interp1(x_tile(:), 1:num_x_tile, OIBc.xq, 'nearest'); % find (sub)index of nearest x-grid coordinate
ky							= interp1(y_tile, 1:num_y_tile, OIBc.yq, 'nearest'); % find (sub)index of nearest y-grid coordinate
ki							= sub2ind([num_y_tile, num_x_tile],ky,kx); % Convert to index of vectorized grid
ku							= unique(ki, 'stable'); % Get unique indices in order
ku(isnan(ku))				= [];
x							= x_grd(ku); % Store unique x-grid coordinates
y							= y_grd(ku); % Store unique y-grid coordinates
thickness					= nan * ones(length(x),1);
for ii = 1:length(ku)
    thickness(ii,1)				= median(OIBc.thickq(ki == ku(ii)), 'omitnan'); % Median of all points with this index  
end
yts				= 60*60*24*365;
thickness = thickness / yts;


save(strcat('/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/GreenlandScape_PINNICLE/',Region,'_xyz_ds.mat'), 'x', 'y', 'thickness', '-v7.3')

clear kx ky ki ku ii OIBf


%% Get basal velocity estimates

Path2dataGS					= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/Data/';
% Get beta
beta						= readGeotiff(strcat(Path2dataGS,'SIA_Results/SIA_Results_slopeflow/GrIS_SIA_beta_150m.tif'));
beta.y						= flipud(beta.y(:));
beta.z						= flipud(beta.z);
% Get mask
Mask						= readGeotiff(strcat(Path2dataGS,'GrIS_BM5_ice_sheet_mask_150m.tif'));
Mask.y						= flipud(Mask.y(:));
mask_gris					= flipud(Mask.z);
% Get velocities
[velx, R]				= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_x_filt_150m.tif'));
vely					= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_y_filt_150m.tif'));
vel						= readgeoraster(strcat(Path2dataGS,'AMC_test/GrIS_Meas_250m_AvgSurfVel_speed_filt_150m.tif'));
velx					= flipud(velx);
vely					= flipud(vely);
vel						= flipud(vel);
velx(~mask_gris)		= NaN;
vely(~mask_gris)		= NaN;
vel(~mask_gris)			= NaN;

% Get PINNICLE data for mesh
Region						= 'UpJak';
PINNICLE_path				= '/Users/achartra/Library/CloudStorage/OneDrive-NASA/Greenland-scape/GreenlandScape_PINNICLE/';
ISSM_run					= 'UpperJakobshavn_issm2025-Jan-17_1';
ISSM_file					= strcat(PINNICLE_path,'Models/', ISSM_run, '.mat');
load(ISSM_file,'md')
md.mesh						= mesh2d(md.mesh);


% Estimate basal velocity
vel_base					= vel .* beta.z;
u_base						= velx .* beta.z;
v_base						= vely .* beta.z;


% Interpolate to mesh
md_u_base = InterpFromGridToMesh(beta.x,beta.y,u_base,md.mesh.x,md.mesh.y,0);
md_v_base = InterpFromGridToMesh(beta.x,beta.y,v_base,md.mesh.x,md.mesh.y,0);


yts				= 60*60*24*365;

md_u_base = md_u_base/yts;
md_v_base = md_v_base/yts;

save(strcat(PINNICLE_path,Region,'_vel_base_ms.mat'),'x','y','md_u_base','md_v_base','-v7.3')