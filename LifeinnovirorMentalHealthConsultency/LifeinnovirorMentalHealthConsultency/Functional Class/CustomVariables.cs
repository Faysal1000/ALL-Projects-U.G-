namespace LifeinnovirorMentalHealthConsultency.Functional_Class
{
    public static class CustomVariables
    {
        public const int loggedSessionValidityForAdminInMinutes = 300;  // 60 min = 1 hour validity
        public const int loggedSessionValidityForDoctorInMinutes = 300; 
        public const int loggedSessionValidityForPatientInMinutes = 300; 
        public const int daysAfterDoctorCanRequestRegistrationAgain = 30;
        public const int maxSizeOfProfilePictureInMB = 5;
        public const string doctorInterviewNotificationMessage = "Your registration request has been reviewed. You are requested to attend an interview. Please check your email or contact admin for further details.";
        public const string doctorRejectNotificationMessage = "We regret to inform you that your doctor registration request has been rejected. For more information, please contact support.";
        public const string doctorApprovedNotificationMessage = "Congratulation. Your registration request is approved";



        //================================================================================
        // Don't change these in a running server
        // Fix these when first setuping the server
        // Otherwise previously saved image will be lost
        public const string temporaryFilePath = "~/App_data/Temp";
        public const string doctorProfilePicturesPath = "~/App_Data/DoctorProfilePhoto";
        public const string patientProfilePicturesPath = "~/App_Data/PatientProfilePhoto";
        //================================================================================
    }
}