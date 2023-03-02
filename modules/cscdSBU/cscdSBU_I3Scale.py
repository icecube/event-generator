from icecube import icetray,dataclasses,phys_services

class I3Scale(icetray.I3Module):

    def __init__(self,context):
        icetray.I3Module.__init__(self,context)
        self.AddParameter("geometry",
                          "Name of I3Geometry to use"
                          "I3Geometry"
                          )
        self.AddParameter("vertex",
                         "Name of vertex to use for calculation",
                         "CredoFit"
                         )
        self.AddParameter("outputname",
                          "Name of resulting double in the frame",
                          "I3XYScale"
                          )
        self.AddParameter("ic_config",
                         "The configuration of the detector, either 'IC79' or 'IC86'",
                         "IC79"
                         )
        self.AddOutBox("OutBox")

    def Configure(self):
        self.geometry      = self.GetParameter("geometry")
        self.vertex        = self.GetParameter("vertex")
        self.outputname    = self.GetParameter("outputname")
        self.ic_config     = self.GetParameter("ic_config")
        
    def Physics(self,frame):

        geo     = frame[self.geometry]
        part    = frame[self.vertex]
        icecubeconfig = None
        if self.ic_config == "IC86":
            icecubeconfig = phys_services.I3ScaleCalculator.IceCubeConfig.IC86
            
        if self.ic_config == "IC79":
            icecubeconfig = phys_services.I3ScaleCalculator.IceCubeConfig.IC79
        if self.ic_config == "G1":
            icecubeconfig = phys_services.I3ScaleCalculator.IceCubeConfig.G1
            
        scale   = phys_services.I3ScaleCalculator(geo,icecubeconfig,phys_services.I3ScaleCalculator.IceTopConfig.IT73)
        result  = scale.scale_inice(part)
        result  = dataclasses.I3Double(result)
        frame[self.outputname] = result

        self.PushFrame(frame)   
        return True        
