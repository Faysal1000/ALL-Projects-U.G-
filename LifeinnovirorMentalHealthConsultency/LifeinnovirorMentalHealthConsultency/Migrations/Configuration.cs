namespace LifeinnovirorMentalHealthConsultency.Migrations
{
    using System;
    using System.Data.Entity;
    using System.Data.Entity.Migrations;
    using System.Linq;
    using LifeinnovirorMentalHealthConsultency.Context.Tables;

    internal sealed class Configuration : DbMigrationsConfiguration<LifeinnovirorMentalHealthConsultency.Context.LifeinnovirorContext>
    {
        public Configuration()
        {
            AutomaticMigrationsEnabled = false;
        }

        protected override void Seed(LifeinnovirorMentalHealthConsultency.Context.LifeinnovirorContext context)
        {
            //  This method will be called after migrating to the latest version.

            //  You can use the DbSet<T>.AddOrUpdate() helper extension method
            //  to avoid creating duplicate seed data.

            bool addDataToDataBase = false;

            if (addDataToDataBase)
            {
                string defaultEmail = "faysalahmmed4200@gmail.com";
                string defaultName = "Faysal Ahmmed";

                // Hash password and security answer
                string hashedPassword = CustomFunctions.GetSha256HashBase64(defaultEmail); // using email as password for simplicity
                string hashedSecurityAnswer = CustomFunctions.GetSha256HashBase64("test"); // simple static answer

                // Seed Admin
                context.Admins.Add(new Admin
                {
                    Email = defaultEmail,
                    FullName = defaultName,
                    PasswordHash = hashedPassword,
                    CreatedAt = DateTime.Now
                });

                // Seed Doctor
                context.Doctors.Add(new Doctor
                {
                    Email = defaultEmail,
                    FullName = defaultName,
                    PhoneNumber = "01700000000",
                    PasswordHash = hashedPassword,
                    SecurityQuestion = "Your pet’s name?",
                    SecurityAnswerHash = hashedSecurityAnswer,
                    Qualifications = "MBBS, FCPS",
                    ExperienceSummary = "5 years of clinical experience",
                    YearsOfExperience = 5,
                    minimumCancelTime = 1,
                    ProfilePhotoUrl = null,
                    Status = "Approved",
                    CreatedAt = DateTime.Now,
                    UpdatedAt = DateTime.Now
                });

                // Seed Patient
                context.Patients.Add(new Patient
                {
                    Email = defaultEmail,
                    FullName = defaultName,
                    PhoneNumber = "01800000000",
                    PasswordHash = hashedPassword,
                    CreatedAt = DateTime.Now,
                    UpdatedAt = DateTime.Now
                    // Optional fields can be left null
                });

                context.SaveChanges();
            }
        }
    }
}
