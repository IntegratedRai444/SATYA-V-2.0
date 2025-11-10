import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Settings as SettingsIcon, Monitor, Bell, Shield, Trash2 } from 'lucide-react';
import { useSettings } from '../hooks/useSettings';
import { useToast } from '../components/ui/use-toast';

export default function Settings() {
  const { settings, isLoading, updateSettings, resetToDefaults } = useSettings();
  const { toast } = useToast();
  
  const handleToggleTheme = async () => {
    try {
      const newTheme = settings.theme === 'dark' ? 'light' : 'dark';
      await updateSettings({ theme: newTheme });
      toast({
        title: 'Theme updated',
        description: `Switched to ${newTheme} mode`,
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to update theme',
        variant: 'destructive',
      });
    }
  };
  
  const handleResetSettings = async () => {
    try {
      await resetToDefaults();
      toast({
        title: 'Settings reset',
        description: 'All settings have been reset to defaults',
      });
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to reset settings',
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="min-h-full bg-bg-primary p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-satyaai-primary/10 rounded-xl flex items-center justify-center">
              <SettingsIcon className="w-6 h-6 text-satyaai-primary" />
            </div>
            <div>
              <h1 className="text-heading-1 text-text-primary font-bold">Settings</h1>
              <p className="text-text-secondary">Manage your SatyaAI preferences and system configuration</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* System Information */}
          <Card className="bg-bg-card border-border-primary">
            <CardHeader>
              <CardTitle className="text-text-primary flex items-center gap-2">
                <Monitor className="w-5 h-5" />
                System Information
              </CardTitle>
              <CardDescription className="text-text-secondary">Current system status and configuration</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium text-text-primary">SatyaAI Version</label>
                <p className="text-sm text-text-secondary">v2.0.0 - Neural Vision</p>
              </div>
              <div>
                <label className="text-sm font-medium text-text-primary">Backend Status</label>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-accent-colors-green rounded-full"></div>
                  <p className="text-sm text-accent-colors-green">Connected</p>
                </div>
              </div>
              <div>
                <label className="text-sm font-medium text-text-primary">Environment</label>
                <p className="text-sm text-text-secondary">Development Mode</p>
              </div>
              <div>
                <label className="text-sm font-medium text-text-primary">Models Loaded</label>
                <p className="text-sm text-satyaai-primary">Neural Vision v4.2</p>
              </div>
            </CardContent>
          </Card>

          {/* Detection Preferences */}
          <Card className="bg-bg-card border-border-primary">
            <CardHeader>
              <CardTitle className="text-text-primary flex items-center gap-2">
                <Shield className="w-5 h-5" />
                Detection Settings
              </CardTitle>
              <CardDescription className="text-text-secondary">Configure analysis parameters and thresholds</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium text-text-primary">Confidence Threshold</label>
                <p className="text-sm text-text-secondary mb-2">Minimum confidence for positive detection</p>
                <div className="flex items-center gap-4">
                  <div className="flex-1 bg-bg-tertiary rounded-lg h-2">
                    <div className="bg-satyaai-primary h-2 rounded-lg" style={{ width: '80%' }}></div>
                  </div>
                  <span className="text-sm font-medium text-satyaai-primary">80%</span>
                </div>
              </div>
              <div>
                <label className="text-sm font-medium text-text-primary">Analysis Mode</label>
                <p className="text-sm text-text-secondary">Comprehensive (Recommended)</p>
              </div>
              <div>
                <label className="text-sm font-medium text-text-primary">Auto-save Results</label>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-accent-colors-green rounded-full"></div>
                  <p className="text-sm text-accent-colors-green">Enabled</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* AI Models Information */}
        <Card className="bg-bg-card border-border-primary">
          <CardHeader>
            <CardTitle className="text-text-primary">AI Models</CardTitle>
            <CardDescription className="text-text-secondary">Information about available detection models</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="border border-border-primary rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-text-primary">XceptionNet</h4>
                <div className="space-y-1 text-sm text-text-secondary">
                  <p>Version: v4.2</p>
                  <p>Type: Image Analysis</p>
                  <p>Accuracy: 98.2%</p>
                </div>
              </div>
              <div className="border border-border-primary rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-text-primary">VideoNet</h4>
                <div className="space-y-1 text-sm text-text-secondary">
                  <p>Version: v3.1</p>
                  <p>Type: Video Analysis</p>
                  <p>Accuracy: 96.8%</p>
                </div>
              </div>
              <div className="border border-border-primary rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-text-primary">AudioNet</h4>
                <div className="space-y-1 text-sm text-text-secondary">
                  <p>Version: v2.5</p>
                  <p>Type: Audio Analysis</p>
                  <p>Accuracy: 95.3%</p>
                </div>
              </div>
              <div className="border border-border-primary rounded-lg p-4">
                <h4 className="font-semibold mb-2 text-text-primary">LiveNet</h4>
                <div className="space-y-1 text-sm text-text-secondary">
                  <p>Version: v1.8</p>
                  <p>Type: Real-time Analysis</p>
                  <p>Accuracy: 92.7%</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Preferences */}
        <Card className="bg-bg-card border-border-primary">
          <CardHeader>
            <CardTitle className="text-text-primary flex items-center gap-2">
              <Bell className="w-5 h-5" />
              Preferences
            </CardTitle>
            <CardDescription className="text-text-secondary">Customize your SatyaAI experience</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-text-primary">Theme</label>
                <p className="text-sm text-text-secondary">{settings.theme} mode is currently active</p>
              </div>
              <Button 
                variant="outline" 
                size="sm" 
                className="border-border-primary text-text-primary hover:bg-bg-tertiary"
                onClick={handleToggleTheme}
                disabled={isLoading}
              >
                Toggle Theme
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-text-primary">Notifications</label>
                <p className="text-sm text-text-secondary">Get notified about analysis results</p>
              </div>
              <Button variant="outline" size="sm" className="border-border-primary text-text-primary hover:bg-bg-tertiary">
                Configure
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-text-primary">Real-time Updates</label>
                <p className="text-sm text-text-secondary">Live analysis status updates</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-accent-colors-green rounded-full"></div>
                <p className="text-sm text-accent-colors-green">Enabled</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Danger Zone */}
        <Card className="bg-bg-card border-accent-colors-red/20">
          <CardHeader>
            <CardTitle className="text-accent-colors-red flex items-center gap-2">
              <Trash2 className="w-5 h-5" />
              Danger Zone
            </CardTitle>
            <CardDescription className="text-text-secondary">Irreversible actions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-text-primary">Clear All History</label>
                <p className="text-sm text-text-secondary">Remove all analysis history from local storage</p>
              </div>
              <Button variant="destructive" size="sm" className="bg-accent-colors-red hover:bg-accent-colors-red/80">
                Clear History
              </Button>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-text-primary">Reset Settings</label>
                <p className="text-sm text-text-secondary">Reset all preferences to default values</p>
              </div>
              <Button 
                variant="destructive" 
                size="sm" 
                className="bg-accent-colors-red hover:bg-accent-colors-red/80"
                onClick={handleResetSettings}
                disabled={isLoading}
              >
                Reset All
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}