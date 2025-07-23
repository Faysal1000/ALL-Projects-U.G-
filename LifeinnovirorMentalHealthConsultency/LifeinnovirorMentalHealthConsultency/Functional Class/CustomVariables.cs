using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Functional_Class
{
    public static class CustomVariables
    {
        public const int loggedSessionValidityForAdminInMinutes = 300; 
        public const int loggedSessionValidityForDoctorInMinutes = 300; 
        public const int loggedSessionValidityForPatientInMinutes = 300; 
        public const int daysAfterDoctorCanRequestRegistrationAgain = 30;
        public const int maxSizeOfProfilePictureInMB = 5;
        public const string temporaryFilePath = "~/App_data/Temp";
        public const string doctorProfilePicturesPath = "~/App_Data/DoctorProfilePhoto";
        public const string patientProfilePicturesPath = "~/App_Data/PatientProfilePhoto";
    }
}