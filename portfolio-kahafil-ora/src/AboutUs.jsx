import AboutKahafilOra from "./components/AboutKahafilOra";
import AboutUsHeroSection from "./components/AboutUsHeroSection";
import LogoAnimation from "./components/LogoAnimation";
import LeadershipRoles from "./components/LeadershipRoles";
import Footer from "./components/Footer";
import MetricMarvels from "./components/MetricMarvels";
import Education from "./components/Education";
import Navbar from "./components/Navbar";

const AboutUs = () => {
  return (
    <div>
      <Navbar />
      <AboutUsHeroSection />
      <AboutKahafilOra />
      <LogoAnimation />
      <Education />
      <LeadershipRoles />
      <MetricMarvels />
      <Footer />
    </div>
  );
};
export default AboutUs;
