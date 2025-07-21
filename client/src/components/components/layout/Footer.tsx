import { Github, Twitter, Linkedin, MessageSquare } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-card border-t border-primary/20 py-4 px-6">
      <div className="flex flex-col md:flex-row justify-between items-center">
        <div className="text-sm text-muted-foreground mb-4 md:mb-0">
          &copy; {new Date().getFullYear()} SatyaAI • Version 1.2.0 •{" "}
          <a href="#" className="text-primary hover:underline">
            Privacy Policy
          </a>
        </div>

        <div className="flex space-x-4">
          <a
            href="#"
            className="text-muted-foreground hover:text-primary transition-colors"
            aria-label="GitHub"
          >
            <Github size={18} />
          </a>
          <a
            href="#"
            className="text-muted-foreground hover:text-primary transition-colors"
            aria-label="Twitter"
          >
            <Twitter size={18} />
          </a>
          <a
            href="#"
            className="text-muted-foreground hover:text-primary transition-colors"
            aria-label="LinkedIn"
          >
            <Linkedin size={18} />
          </a>
          <a
            href="#"
            className="text-muted-foreground hover:text-primary transition-colors"
            aria-label="Discord"
          >
            <MessageSquare size={18} />
          </a>
        </div>
      </div>
    </footer>
  );
}
