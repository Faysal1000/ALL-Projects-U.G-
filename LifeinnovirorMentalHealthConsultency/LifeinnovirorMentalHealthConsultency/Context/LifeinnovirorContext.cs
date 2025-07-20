using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Data.Entity.ModelConfiguration.Conventions;
using System.Linq;
using System.Web;
using LifeinnovirorMentalHealthConsultency.Context.Tables;

namespace LifeinnovirorMentalHealthConsultency.Context
{
    public class LifeinnovirorContext: DbContext
    {
        public LifeinnovirorContext() : base("name=LifeinnovirorContext") { }

        public LifeinnovirorContext(string connectionString) : base(connectionString) { }
        public DbSet<Admin> Admins { get; set; }
        public DbSet<Doctor> Doctors { get; set; }
        public DbSet<Patient> Patients { get; set; }
        public DbSet<Specialization> Specializations { get; set; }
        public DbSet<SubSpecialization> SubSpecializations { get; set; }
        public DbSet<DoctorSubSpecialization> DoctorSubSpecializations { get; set; }
        public DbSet<Language> Languages { get; set; }
        public DbSet<DoctorLanguage> DoctorLanguages { get; set; }
        public DbSet<Certificate> Certificates { get; set; }
        public DbSet<FAQ> FAQs { get; set; }
        public DbSet<RegistrationRequest> RegistrationRequests { get; set; }
        public DbSet<AppointmentType> AppointmentTypes { get; set; }
        public DbSet<DoctorTimeSlot> DoctorTimeSlots { get; set; }
        public DbSet<Appointment> Appointments { get; set; }
        public DbSet<Payment> Payments { get; set; }
        public DbSet<LeaveRequest> LeaveRequests { get; set; }
        public DbSet<DoctorStatusHistory> DoctorStatusHistories { get; set; }
        public DbSet<SecureChatMessage> SecureChatMessages { get; set; }
        public DbSet<Report> Reports { get; set; }
        public DbSet<Feedback> Feedbacks { get; set; }
        public DbSet<SystemLog> SystemLogs { get; set; }
        public DbSet<Notification> Notifications { get; set; }
        protected override void OnModelCreating(DbModelBuilder modelBuilder)
        {
            // Composite key for DoctorSubSpecialization
            modelBuilder.Entity<DoctorSubSpecialization>()
                .HasKey(ds => new { ds.DoctorId, ds.SubSpecializationId });

            // Composite key for DoctorLanguage
            modelBuilder.Entity<DoctorLanguage>()
                .HasKey(dl => new { dl.DoctorId, dl.LanguageId });

            // Turn off cascade delete for Doctor → DoctorTimeSlots
            modelBuilder.Entity<DoctorTimeSlot>()
                .HasRequired(dts => dts.Doctor)
                .WithMany(d => d.DoctorTimeSlots)
                .HasForeignKey(dts => dts.DoctorId)
                .WillCascadeOnDelete(false);

            base.OnModelCreating(modelBuilder);
        }
    }
}