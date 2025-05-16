import { useState } from "react";
import { Helmet } from 'react-helmet';
import { Check, Sun, Moon, Languages } from "lucide-react";
import { useTheme } from "next-themes";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export default function Settings() {
  const { theme, setTheme } = useTheme();
  const { toast } = useToast();
  
  // Preferences state
  const [language, setLanguage] = useState("english");
  const [confidenceThreshold, setConfidenceThreshold] = useState([75]);
  const [enableNotifications, setEnableNotifications] = useState(true);
  const [autoAnalyze, setAutoAnalyze] = useState(true);
  const [sensitivityLevel, setSensitivityLevel] = useState("medium");
  
  // Profile state
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  
  // Handle save preferences
  const { mutate: savePreferences, isPending: isSavingPreferences } = useMutation({
    mutationFn: async () => {
      return await apiRequest('POST', '/api/settings/preferences', {
        language,
        confidenceThreshold: confidenceThreshold[0],
        enableNotifications,
        autoAnalyze,
        sensitivityLevel,
        theme
      });
    },
    onSuccess: () => {
      toast({
        title: "Preferences saved",
        description: "Your preferences have been updated successfully."
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error saving preferences",
        description: error.message,
        variant: "destructive"
      });
    }
  });
  
  // Handle save profile
  const { mutate: saveProfile, isPending: isSavingProfile } = useMutation({
    mutationFn: async () => {
      return await apiRequest('POST', '/api/settings/profile', {
        email,
        name
      });
    },
    onSuccess: () => {
      toast({
        title: "Profile saved",
        description: "Your profile has been updated successfully."
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Error saving profile",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  return (
    <>
      <Helmet>
        <title>SatyaAI - Settings</title>
        <meta name="description" content="Customize your SatyaAI experience. Adjust theme, detection thresholds, notification preferences, and manage your account settings." />
      </Helmet>
      
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Settings</h1>
        <p className="text-muted-foreground">
          Customize your experience with SatyaAI detection system.
        </p>
      </div>
      
      <Tabs defaultValue="preferences">
        <TabsList className="grid w-full grid-cols-2 mb-6">
          <TabsTrigger value="preferences">Preferences</TabsTrigger>
          <TabsTrigger value="profile">Profile & Account</TabsTrigger>
        </TabsList>
        
        <TabsContent value="preferences">
          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Appearance</CardTitle>
                <CardDescription>
                  Customize how SatyaAI looks and feels.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Theme Mode</Label>
                    <div className="text-sm text-muted-foreground">
                      Choose between light and dark mode.
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant={theme === "light" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setTheme("light")}
                      className="w-24"
                    >
                      <Sun className="mr-2 h-4 w-4" />
                      Light
                    </Button>
                    <Button
                      variant={theme === "dark" ? "default" : "outline"}
                      size="sm"
                      onClick={() => setTheme("dark")}
                      className="w-24"
                    >
                      <Moon className="mr-2 h-4 w-4" />
                      Dark
                    </Button>
                  </div>
                </div>
                
                <div className="flex flex-row items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Language</Label>
                    <div className="text-sm text-muted-foreground">
                      Select your preferred language.
                    </div>
                  </div>
                  <Select value={language} onValueChange={setLanguage}>
                    <SelectTrigger className="w-40">
                      <div className="flex items-center gap-2">
                        <Languages className="h-4 w-4" />
                        <SelectValue placeholder="Language" />
                      </div>
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="english">English</SelectItem>
                      <SelectItem value="spanish">Spanish</SelectItem>
                      <SelectItem value="french">French</SelectItem>
                      <SelectItem value="german">German</SelectItem>
                      <SelectItem value="japanese">Japanese</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Detection Settings</CardTitle>
                <CardDescription>
                  Customize the sensitivity and thresholds for deepfake detection.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Confidence Threshold</Label>
                    <span className="text-sm font-medium">{confidenceThreshold[0]}%</span>
                  </div>
                  <Slider
                    defaultValue={[75]}
                    max={100}
                    min={50}
                    step={1}
                    value={confidenceThreshold}
                    onValueChange={setConfidenceThreshold}
                  />
                  <p className="text-muted-foreground text-sm">
                    Media with confidence score above this threshold will be flagged as deepfake.
                  </p>
                </div>
                
                <div className="space-y-2">
                  <Label>Detection Sensitivity</Label>
                  <Select value={sensitivityLevel} onValueChange={setSensitivityLevel}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select sensitivity" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low (Fewer false positives)</SelectItem>
                      <SelectItem value="medium">Medium (Balanced)</SelectItem>
                      <SelectItem value="high">High (Catch more potential deepfakes)</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-muted-foreground text-sm">
                    Higher sensitivity may increase false positives but catch more subtle manipulations.
                  </p>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="auto-analyze">Auto Analyze Uploads</Label>
                    <div className="text-sm text-muted-foreground">
                      Automatically start analysis when media is uploaded.
                    </div>
                  </div>
                  <Switch
                    id="auto-analyze"
                    checked={autoAnalyze}
                    onCheckedChange={setAutoAnalyze}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="notifications">Enable Notifications</Label>
                    <div className="text-sm text-muted-foreground">
                      Receive notifications when analysis is complete.
                    </div>
                  </div>
                  <Switch
                    id="notifications"
                    checked={enableNotifications}
                    onCheckedChange={setEnableNotifications}
                  />
                </div>
              </CardContent>
              <CardFooter>
                <Button 
                  onClick={() => savePreferences()}
                  disabled={isSavingPreferences}
                  className="ml-auto"
                >
                  {isSavingPreferences ? "Saving..." : "Save Preferences"}
                </Button>
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="profile">
          <Card>
            <CardHeader>
              <CardTitle>Profile Information</CardTitle>
              <CardDescription>
                Update your account details and preferences.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <Input 
                  id="name" 
                  value={name} 
                  onChange={(e) => setName(e.target.value)} 
                  placeholder="Your Name"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input 
                  id="email" 
                  type="email" 
                  value={email} 
                  onChange={(e) => setEmail(e.target.value)} 
                  placeholder="your.email@example.com"
                />
              </div>
              
              <div className="pt-4">
                <h3 className="font-semibold mb-2">Connected Services</h3>
                <div className="rounded-md border p-4">
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="font-medium">SatyaAI Cloud</p>
                      <p className="text-sm text-muted-foreground">
                        Sync your scan history across devices
                      </p>
                    </div>
                    <Button variant="outline" size="sm" className="flex items-center gap-1">
                      <Check className="h-4 w-4 text-green-500" />
                      Connected
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline">Reset to Defaults</Button>
              <Button 
                onClick={() => saveProfile()}
                disabled={isSavingProfile}
              >
                {isSavingProfile ? "Saving..." : "Save Profile"}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </>
  );
}
