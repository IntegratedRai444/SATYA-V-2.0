import { Link, useLocation } from "wouter";
import { Menu, Bell, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface HeaderProps {
  toggleSidebar: () => void;
}

export default function Header({ toggleSidebar }: HeaderProps) {
  const [location] = useLocation();

  const navItems = [
    { name: "Home", path: "/" },
    { name: "Scan", path: "/scan" },
    { name: "History", path: "/history" },
    { name: "Settings", path: "/settings" },
    { name: "Help", path: "/help" },
  ];

  return (
    <header className="bg-card border-b border-primary/30 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-4">
        {/* Logo and Brand */}
        <div className="flex items-center">
          <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center mr-3">
            <span className="text-primary text-2xl font-bold">S</span>
          </div>
          <h1 className="text-2xl font-bold text-foreground font-poppins">
            Satya<span className="text-primary">AI</span>
          </h1>
        </div>
        <p className="text-muted-foreground text-sm hidden md:block">
          Synthetic Authentication Technology for Your Analysis
        </p>
      </div>

      {/* Navigation */}
      <nav className="hidden md:flex">
        <ul className="flex space-x-6">
          {navItems.map((item) => (
            <li key={item.path}>
              <Link href={item.path}>
                <a
                  className={cn(
                    "font-poppins transition-colors",
                    location === item.path
                      ? "text-primary font-medium"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {item.name}
                </a>
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      {/* Profile and Mobile Menu */}
      <div className="flex items-center">
        <Button variant="ghost" size="icon" className="rounded-full mr-2">
          <Bell className="h-5 w-5 text-muted-foreground" />
        </Button>
        <div className="w-9 h-9 rounded-full bg-primary/20 flex items-center justify-center cursor-pointer">
          <span className="text-primary font-bold">U</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden ml-4"
          onClick={toggleSidebar}
        >
          <Menu className="h-5 w-5 text-muted-foreground" />
        </Button>
      </div>
    </header>
  );
}
