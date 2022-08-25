using DelimitedFiles, Copulas

include("src/bivarcopula_fitting.jl")
include("src/bivarcopula_functions.jl")

lonlat = readdlm("data/lonlat.csv") #gridbox centers
data = readdlm("data/data.csv") #time series

empcopdata = Copulas.pseudos(data) #empirical cdf transformation to unifority, could also estimate BATs and apply cdf (results were similar in either case)

# now we reproduce an example fit in the paper
# which shows strong opposite tail dependence of a blocking extreme (warm in Alaska, cold in midlatitudes of North Ammerica)
loc1index = 2
loc2index = 49 
# try loc1index = 72 and loc2index = 65 for strong positive dependence
initeven_gumbel = ([1.5; 1.5; 1.5; 1.5; 0.25; 0.25; 0.25],[2.0; 2.0; 2.0; 2.0; 0.25; 0.25; 0.25]) #two initial guesses
examplefit = fit_Rotated_BivariateCopula_robust(empcopdata[loc1index,:], empcopdata[loc2index,:], initeven_gumbel, :gumbel, num_attempts=100, print_level=5) 

# plotting the log density of the estimated mixture model
ustepsize = 0.01
ugridbaseseq = ustepsize:ustepsize:(1.0-ustepsize)
ugrid = Matrix(reshape(reinterpret(Float64,vec(collect(Iterators.product(ugridbaseseq, ugridbaseseq)))), (2,:)))

mle = [x for x in examplefit.mle]
pdfgrid = pdf_Rotated_BivariateCopula(ugrid[1,:],ugrid[2,:], mle, :gumbel)

using RCall
R"""
library(ggplot2)
library(RColorBrewer)
rdbucolors = rev(brewer.pal(n=5, name = "RdBu"))
opppairlonlat1 = $lonlat[$loc1index,]
opppairlonlat2 = $lonlat[$loc2index,]
df = data.frame(x = $ugrid[1,], y = $ugrid[2,], logpdf = log($pdfgrid))
scaleqvalues = quantile(c(df$logpdf),na.rm=TRUE, probs = c(0,.01,.999,1))
scalevalues = scales::rescale(c(scaleqvalues[1:2], scaleqvalues[2], -0.1, -0.1, 0.1, 0.1, scaleqvalues[3], scaleqvalues[3:4]))
ggplot(data = df) + theme_bw() + geom_raster(aes(x = x, y = y, fill = logpdf)) + scale_fill_gradientn(colours = rdbucolors, na.value = NA, values = scalevalues, limits=range(scaleqvalues)) + coord_fixed(ratio = 1) + scale_x_continuous(expand=c(0,0),limits=c(0,1)) + scale_y_continuous(expand=c(0,0),limits=c(0,1)) + labs("colour = log\ndensity", fill="log\ndensity") + theme(legend.key.size = unit(3,"line"), legend.text=element_text(size=20), legend.title=element_text(size=20), plot.title = element_text(hjust=0.5, size=25), axis.title.x=element_text(size=25), axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.y=element_text(size=25), axis.text.y=element_blank(), axis.ticks.y=element_blank(), plot.background = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank()) + xlab(paste0("Lon = ",opppairlonlat1[1],", Lat = ",opppairlonlat1[2])) + ylab(paste0("Lon = ",opppairlonlat2[1],", Lat = ",opppairlonlat2[2])) 
"""