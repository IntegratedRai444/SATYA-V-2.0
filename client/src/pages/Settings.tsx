import { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/components/ui/use-toast';
import { FiUser, FiCamera, FiLock, FiTrash2, FiSave } from 'react-icons/fi';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import apiClient from '@/lib/api';

export default function Settings() {
  const { user, logout } = useAuth();
  const { toast } = useToast();
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [profileData, setProfileData] = useState({
    fullName: user?.fullName || '',
    email: user?.email || '',
    username: user?.username || '',
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  // Load profile data on mount
  useEffect(() => {
    if (user) {
      setProfileData({
        fullName: user.fullName || '',
        email: user.email || '',
        username: user.username || '',
      });
    }
  }, [user]);

  const handleSaveProfile = async () => {
    setIsLoading(true);
    try {
      const response = await apiClient.updateProfile({
        fullName: profileData.fullName,
        email: profileData.email,
      });

      if (response.success) {
        toast({
          title: 'Profile Updated',
          description: 'Your profile has been saved successfully.',
        });
        setIsEditing(false);
      } else {
        toast({
          title: 'Error',
          description: response.message || 'Failed to update profile',
          variant: 'destructive',
        });
      }
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to update profile',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };


  const handleChangePassword = async () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      toast({
        title: 'Error',
        description: 'Passwords do not match',
        variant: 'destructive',
      });
      return;
    }

    if (passwordData.newPassword.length < 8) {
      toast({
        title: 'Error',
        description: 'Password must be at least 8 characters',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await apiClient.changePassword(
        passwordData.currentPassword,
        passwordData.newPassword
      );

      if (response.success) {
        toast({
          title: 'Password Changed',
          description: 'Your password has been updated successfully.',
        });
        setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
      } else {
        toast({
          title: 'Error',
          description: response.message || 'Failed to change password',
          variant: 'destructive',
        });
      }
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to change password',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    const confirmed = window.confirm(
      'Are you sure you want to delete your account? This action cannot be undone.'
    );

    if (!confirmed) return;

    setIsLoading(true);
    try {
      const response = await apiClient.deleteAccount();

      if (response.success) {
        toast({
          title: 'Account Deleted',
          description: 'Your account has been permanently deleted.',
        });
        await logout();
      } else {
        toast({
          title: 'Error',
          description: response.message || 'Failed to delete account',
          variant: 'destructive',
        });
      }
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to delete account',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getUserInitial = () => {
    if (!user) return 'U';
    const name = user.fullName || user.username || 'User';
    return name.charAt(0).toUpperCase();
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-6 min-h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="mb-6 sm:mb-8">
        <h1 className="text-[24px] sm:text-[28px] font-bold text-white mb-2">Profile Settings</h1>
        <p className="text-gray-400 text-[13px] sm:text-[14px]">Manage your account information</p>
      </div>

      {/* Profile Card */}
      <Card className="bg-[#0f1419] border border-gray-800/50 p-4 sm:p-6 lg:p-8 mb-4 sm:mb-6">
        {/* Profile Picture */}
        <div className="flex flex-col sm:flex-row items-center gap-4 sm:gap-6 mb-6 sm:mb-8 pb-6 sm:pb-8 border-b border-gray-800">
          <div className="relative flex-shrink-0">
            <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-full bg-cyan-500 flex items-center justify-center text-white font-bold text-2xl sm:text-3xl shadow-lg shadow-cyan-500/30">
              {getUserInitial()}
            </div>
            <button className="absolute bottom-0 right-0 w-7 h-7 sm:w-8 sm:h-8 bg-gray-800 hover:bg-gray-700 rounded-full flex items-center justify-center border-2 border-[#0f1419] transition-colors">
              <FiCamera className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-300" />
            </button>
          </div>
          <div className="text-center sm:text-left">
            <h3 className="text-white font-semibold text-[15px] sm:text-[16px] mb-1">{profileData.fullName || 'User'}</h3>
            <p className="text-gray-400 text-[12px] sm:text-[13px] mb-3">{profileData.email}</p>
            <div className="flex flex-wrap gap-2 justify-center sm:justify-start">
              <Button className="bg-gray-800 hover:bg-gray-700 text-white text-[11px] sm:text-[12px] h-7 sm:h-8 px-2.5 sm:px-3" disabled>
                Change Photo
              </Button>
              <Button className="bg-transparent hover:bg-gray-800/50 text-gray-400 text-[11px] sm:text-[12px] h-7 sm:h-8 px-2.5 sm:px-3 border border-gray-800" disabled>
                Remove
              </Button>
            </div>
          </div>
        </div>

        {/* Profile Form */}
        <div className="space-y-4 sm:space-y-5">
          <div>
            <label className="block text-[12px] sm:text-[13px] font-medium text-gray-300 mb-2">
              Full Name
            </label>
            <input
              type="text"
              value={profileData.fullName}
              onChange={(e) => setProfileData({ ...profileData, fullName: e.target.value })}
              disabled={!isEditing || isLoading}
              className="w-full px-3 sm:px-4 py-2 sm:py-2.5 bg-[#1a1a1a] border border-gray-800 rounded-lg text-white text-[13px] sm:text-[14px] focus:border-cyan-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              placeholder="Enter your full name"
            />
          </div>

          <div>
            <label className="block text-[12px] sm:text-[13px] font-medium text-gray-300 mb-2">
              Email Address
            </label>
            <input
              type="email"
              value={profileData.email}
              onChange={(e) => setProfileData({ ...profileData, email: e.target.value })}
              disabled={!isEditing || isLoading}
              className="w-full px-3 sm:px-4 py-2 sm:py-2.5 bg-[#1a1a1a] border border-gray-800 rounded-lg text-white text-[13px] sm:text-[14px] focus:border-cyan-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              placeholder="your.email@example.com"
            />
          </div>

          <div>
            <label className="block text-[12px] sm:text-[13px] font-medium text-gray-300 mb-2">
              Username
            </label>
            <input
              type="text"
              value={profileData.username}
              disabled
              className="w-full px-3 sm:px-4 py-2 sm:py-2.5 bg-[#1a1a1a] border border-gray-800 rounded-lg text-gray-500 text-[13px] sm:text-[14px] cursor-not-allowed"
            />
            <p className="text-[10px] sm:text-[11px] text-gray-500 mt-1.5">Username cannot be changed</p>
          </div>

          <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 pt-3 sm:pt-4">
            {!isEditing ? (
              <Button
                onClick={() => setIsEditing(true)}
                className="bg-cyan-500 hover:bg-cyan-600 text-white"
                disabled={isLoading}
              >
                <FiUser className="w-4 h-4 mr-2" />
                Edit Profile
              </Button>
            ) : (
              <>
                <Button
                  onClick={handleSaveProfile}
                  className="bg-cyan-500 hover:bg-cyan-600 text-white"
                  disabled={isLoading}
                >
                  <FiSave className="w-4 h-4 mr-2" />
                  {isLoading ? 'Saving...' : 'Save Changes'}
                </Button>
                <Button
                  onClick={() => setIsEditing(false)}
                  className="bg-gray-800 hover:bg-gray-700 text-white"
                  disabled={isLoading}
                >
                  Cancel
                </Button>
              </>
            )}
          </div>
        </div>
      </Card>

      {/* Change Password Card */}
      <Card className="bg-[#0f1419] border border-gray-800/50 p-4 sm:p-6 lg:p-8 mb-4 sm:mb-6">
        <div className="flex items-center gap-2 sm:gap-3 mb-4 sm:mb-6">
          <FiLock className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400" />
          <h2 className="text-[16px] sm:text-[18px] font-bold text-white">Change Password</h2>
        </div>

        <div className="space-y-3 sm:space-y-4">
          <div>
            <label className="block text-[12px] sm:text-[13px] font-medium text-gray-300 mb-2">
              Current Password
            </label>
            <input
              type="password"
              value={passwordData.currentPassword}
              onChange={(e) => setPasswordData({ ...passwordData, currentPassword: e.target.value })}
              disabled={isLoading}
              className="w-full px-3 sm:px-4 py-2 sm:py-2.5 bg-[#1a1a1a] border border-gray-800 rounded-lg text-white text-[13px] sm:text-[14px] focus:border-cyan-500 focus:outline-none transition-colors disabled:opacity-50"
              placeholder="Enter current password"
            />
          </div>

          <div>
            <label className="block text-[12px] sm:text-[13px] font-medium text-gray-300 mb-2">
              New Password
            </label>
            <input
              type="password"
              value={passwordData.newPassword}
              onChange={(e) => setPasswordData({ ...passwordData, newPassword: e.target.value })}
              disabled={isLoading}
              className="w-full px-3 sm:px-4 py-2 sm:py-2.5 bg-[#1a1a1a] border border-gray-800 rounded-lg text-white text-[13px] sm:text-[14px] focus:border-cyan-500 focus:outline-none transition-colors disabled:opacity-50"
              placeholder="Enter new password"
            />
          </div>

          <div>
            <label className="block text-[12px] sm:text-[13px] font-medium text-gray-300 mb-2">
              Confirm New Password
            </label>
            <input
              type="password"
              value={passwordData.confirmPassword}
              onChange={(e) => setPasswordData({ ...passwordData, confirmPassword: e.target.value })}
              disabled={isLoading}
              className="w-full px-3 sm:px-4 py-2 sm:py-2.5 bg-[#1a1a1a] border border-gray-800 rounded-lg text-white text-[13px] sm:text-[14px] focus:border-cyan-500 focus:outline-none transition-colors disabled:opacity-50"
              placeholder="Confirm new password"
            />
          </div>

          <Button
            onClick={handleChangePassword}
            className="bg-cyan-500 hover:bg-cyan-600 text-white"
            disabled={isLoading || !passwordData.currentPassword || !passwordData.newPassword || !passwordData.confirmPassword}
          >
            {isLoading ? 'Updating...' : 'Update Password'}
          </Button>
        </div>
      </Card>

      {/* Delete Account */}
      <Card className="bg-[#0f1419] border border-gray-800/50 p-4 sm:p-6 lg:p-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-[14px] sm:text-[15px] font-medium text-white mb-1">Delete Account</h3>
            <p className="text-[11px] sm:text-[12px] text-gray-400">Permanently delete your account and all data</p>
          </div>
          <Button
            onClick={handleDeleteAccount}
            className="bg-transparent hover:bg-red-500/10 text-gray-400 hover:text-red-400 text-[12px] sm:text-[13px] border border-gray-800 hover:border-red-500/30 transition-colors w-full sm:w-auto flex-shrink-0"
            disabled={isLoading}
          >
            <FiTrash2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 mr-2" />
            Delete Account
          </Button>
        </div>
      </Card>
    </div>
  );
}
