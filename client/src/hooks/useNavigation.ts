import { useLocation } from "wouter";

/**
 * A custom hook that creates a navigate function similar to other router libraries
 * but using wouter's useLocation hook under the hood
 */
export function useNavigation() {
  const [, setLocation] = useLocation();
  
  // A function to navigate to a new path
  const navigate = (to: string) => {
    setLocation(to);
  };
  
  return { navigate };
}